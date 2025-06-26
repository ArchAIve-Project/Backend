import torch
from transformers import (
    BertTokenizerFast, BertConfig,
    BertForTokenClassification
)
from addons import ModelStore, ASTracer, ASReport, ArchSmith

# ======== NER Pipeline ========
class NERPipeline:
    LABEL_LIST = [
        'B-ACT', 'B-DATE', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PERSON',
        'I-ACT', 'I-DATE', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PERSON', 'O'
    ]
    MODEL_NAME = "bert-base-cased"
    id2label = None
    label2id = None
    tokenizer = None

    @staticmethod
    def load_label_maps():
        NERPipeline.label2id = {label: i for i, label in enumerate(NERPipeline.LABEL_LIST)}
        NERPipeline.id2label = {i: label for i, label in enumerate(NERPipeline.LABEL_LIST)}

    @staticmethod
    def load_model(model_path: str):
        NERPipeline.load_label_maps()
        NERPipeline.tokenizer = BertTokenizerFast.from_pretrained(NERPipeline.MODEL_NAME)

        config = BertConfig.from_pretrained(
            NERPipeline.MODEL_NAME,
            num_labels=len(NERPipeline.label2id),
            label2id=NERPipeline.label2id,
            id2label=NERPipeline.id2label
        )
        model = BertForTokenClassification(config)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict.get("model_state_dict", state_dict))
        model.eval()

        return model

    @staticmethod
    def extract_entities(pairs):
        entities = []
        current_entity = ""
        current_type = None

        for word, tag in pairs:
            if tag == "O":
                if current_entity:
                    entities.append((current_entity.strip(), current_type))
                    current_entity = ""
                    current_type = None
            elif tag.startswith("B-"):
                if current_entity:
                    entities.append((current_entity.strip(), current_type))
                current_entity = word
                current_type = tag[2:]
            elif tag.startswith("I-"):
                if current_type == tag[2:]:
                    current_entity += " " + word
                else:
                    # Handle improper I- without B-
                    if current_entity:
                        entities.append((current_entity.strip(), current_type))
                    current_entity = word
                    current_type = tag[2:]

        if current_entity:
            entities.append((current_entity.strip(), current_type))

        return entities

    @staticmethod
    def predict(sentence: str, tracer: ASTracer):
        inputs = NERPipeline.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
        word_ids = inputs.word_ids(batch_index=0)

        ## Get context object from ModelStore
        model_ctx = ModelStore.getModel("ner")
        if model_ctx is None or model_ctx.model is None:
            return "ERROR: NER model not loaded."
        
        model = model_ctx.model

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        aligned_preds = []
        previous_word_id = None
        for pred, word_id in zip(predictions, word_ids):
            if word_id is None or word_id == previous_word_id:
                continue
            label = NERPipeline.id2label[pred]
            aligned_preds.append(label)
            previous_word_id = word_id

        words = sentence.split()
        result = list(zip(words, aligned_preds))

        report = ASReport(
            source="NERPipeline",
            message=f"Predicted: {len(result)} entities",
            extraData={"sentence": sentence}
        )
        tracer.addReport(report)

        # Extract entities from the result
        entities = NERPipeline.extract_entities(result)

        return entities

if __name__ == "__main__":
    ModelStore.setup(ner=NERPipeline.load_model)

    tracer = ArchSmith.newTracer("NER testing :")

    sentence = "Jun Han was climbing the rock on the The Great Wall Of China"
    
    _, entities = NERPipeline.predict(
        sentence=sentence,
        tracer=tracer
    )
    
    print(entities)

    tracer.end()
    ArchSmith.persist()