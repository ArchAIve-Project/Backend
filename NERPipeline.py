import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from addons import ModelStore, ASTracer, ASReport, ArchSmith

# ======== NER Pipeline ========
class NERPipeline:    
    loaded = False
    MODEL_NAME = "boltuix/EntityBERT"
    id2label = None
    label2id = None
    tokenizer = None

    @staticmethod
    def load_model(model_path: str = None):
        NERPipeline.tokenizer = AutoTokenizer.from_pretrained(NERPipeline.MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(NERPipeline.MODEL_NAME)
        NERPipeline.id2label = model.config.id2label
        NERPipeline.label2id = model.config.label2id
        model.eval()
        NERPipeline.loaded = True
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
                    if current_entity:
                        entities.append((current_entity.strip(), current_type))
                    current_entity = word
                    current_type = tag[2:]
        
        if current_entity:
            entities.append((current_entity.strip(), current_type))
        
        return entities
    
    @staticmethod
    def get_words_and_labels(inputs, predictions, word_ids, id2label):
        words = []
        labels = []

        current_word = ""
        current_label = None
        previous_word_id = None

        tokens = NERPipeline.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        for token, pred_label_id, word_id in zip(tokens, predictions, word_ids):
            if word_id is None:
                continue 

            label = id2label[pred_label_id]

            if word_id != previous_word_id:
                # New word
                if current_word:
                    words.append(current_word)
                    labels.append(current_label)

                if token.startswith("Ġ"):
                    current_word = token[1:]
                else:
                    current_word = token

                current_label = label
                previous_word_id = word_id
            else:
                # Continuation of the same word
                current_word += token

        # Add last word
        if current_word:
            words.append(current_word)
            labels.append(current_label)

        return list(zip(words, labels))

    @staticmethod
    def predict(sentence: str, tracer: ASTracer):
        if not NERPipeline.loaded:
            return "ERROR: NER pipeline has not been loaded. Call load_model() first."

        inputs = NERPipeline.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
        word_ids = inputs.word_ids(batch_index=0)

        model_ctx = ModelStore.getModel("ner")
        if model_ctx is None or model_ctx.model is None:
            return "ERROR: NER model not loaded."
        
        model = model_ctx.model

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        # Use get_words_and_labels to get aligned words and labels
        word_label_pairs = NERPipeline.get_words_and_labels(inputs, predictions, word_ids, NERPipeline.id2label)

        report = ASReport(
            source="NERPipeline",
            message=f"Predicted: {len(word_label_pairs)} tokens",
            extraData={"sentence": sentence}
        )
        tracer.addReport(report)

        # Extract entities from the aligned (word, label) pairs
        entities = NERPipeline.extract_entities(word_label_pairs)

        return entities


if __name__ == "__main__":
    ModelStore.setup(ner=NERPipeline.load_model)

    tracer = ArchSmith.newTracer("NER testing :")

    sentence = """
The original regulations of this Chamber of Commerce were
mailed out on the tenth day of October, designating it as the date for discussion during the winter
solstice meeting. Because gathering on that day was inconvenient, a new date and venue were
set.\n\nAt two o'clock in the afternoon on the tenth day of October, the assembly convened to
deliberate upon the following important matters:\n\n1. The location for holding meetings must be
officially announced beforehand; the procedures for commercial affairs management must be clearly
defined.\n2. Issues including commissions at the meeting place and uncollected taxes should be
publicly displayed and categorized as routine business.\n3. Regarding personnel appointments within
the Chamber's council, any changes or adjustments shall be finalized by the twentieth day of next
year's twelfth lunar month.\n4. Advance voting for elections must be completed ten days prior to the
annual general meeting, which is scheduled for the designated week containing the tenth day of the
tenth lunar month, with official notices to be issued accordingly.\n5. Whether candidates are
qualified for council membership requires deliberation and confirmation by existing members.\n6. A
revised draft outlining operational rules and regulations needs to be discussed and approved.\n7. On
this day, it was resolved that businesses seeking registration but facing practical difficulties
should formally submit petitions through the Chamber.\n8. Mr. Huang Zhanjie, Lin Huá, and Wang Bing
jointly submitted proposals regarding registration exemptions and related matters.\n9. Additionally,
it was proposed that public commissions and major shipping fees currently under consideration should
either be uniformly standardized or temporarily suspended until further study.\n10. Given the
complexity of these issues, it was agreed that the Standing Committee should conduct thorough
investigations and prepare concrete proposals for future deliberation.\n11. Furthermore, discussions
addressed whether members of the comprador association and other entities not yet formally
registered should immediately apply for official recognition according to current bylaws, despite
existing implementation challenges.
"""
    
    entities = NERPipeline.predict(
        sentence=sentence,
        tracer=tracer
    )
    
    print(entities)

    tracer.end()
    ArchSmith.persist()