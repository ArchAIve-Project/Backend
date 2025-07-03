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
                continue  # skip special tokens like CLS, SEP

            label = id2label[pred_label_id]

            if word_id != previous_word_id:
                if current_word:
                    words.append(current_word)
                    labels.append(current_label)

                # Remove BERT-style subword prefix "##"
                current_word = token.replace("##", "")
                current_label = label
                previous_word_id = word_id
            else:
                # Continuation of the same word
                if token.startswith("##"):
                    current_word += token[2:]
                else:
                    current_word += token

        if current_word:
            words.append(current_word)
            labels.append(current_label)

        return list(zip(words, labels))

    @staticmethod
    def predict(sentence: str, tracer: ASTracer):
        if not NERPipeline.loaded:
            return "ERROR: NER pipeline has not been loaded. Call load_model() first."

        inputs = NERPipeline.tokenizer(sentence, return_tensors="pt", truncation=False, is_split_into_words=False)
        word_ids = inputs.word_ids(batch_index=0)

        model_ctx = ModelStore.getModel("ner")
        if model_ctx is None or model_ctx.model is None:
            return "ERROR: NER model not loaded."

        model = model_ctx.model

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        word_label_pairs = NERPipeline.get_words_and_labels(inputs, predictions, word_ids, NERPipeline.id2label)

        tracer.addReport(ASReport(
            source="NERPipeline",
            message=f"Predicted: {len(word_label_pairs)} tokens",
            extraData={"sentence": sentence}
        ))

        # Extract all entities
        all_entities = NERPipeline.extract_entities(word_label_pairs)

        # Filter out CARDINAL entities
        entities = [ent for ent in all_entities if ent[1] != "CARDINAL"]

        return entities


if __name__ == "__main__":
    ModelStore.setup(ner=NERPipeline.load_model)

    tracer = ArchSmith.newTracer("NER testing :")

    sentence = """
    The original bylaws of this Chamber stipulated October 10th as the date for the winter session, but due to the Empress Dowager's birthday celebration, it was rescheduled with special arrangement.
    On the tenth day of the eleventh lunar month at two o'clock in the afternoon, a meeting will convene to discuss important matters listed below:

    1. To deliberate on the procedures proposed by the government of Singapore regarding commercial registration.
    2. To consider whether the public notice delineating the nine-eight commission and large vessel taxes in Singapore should be implemented.
    3. To elect the members of the Chamber’s committee for next year, with nominations beginning on the tenth day of the first lunar month and concluding with the annual meeting on April 10th.
    4. To determine if individuals are eligible to serve as committee members.
    5. To review the administrative charter concerning merchants returning to their hometowns. During the meeting, it was resolved that the commercial registration requirements were inconvenient for merchants and should be addressed through a petition submitted by the Chamber to the British authorities. The draft of the petition was prepared by three gentlemen: Lin Bingxiang, Lin Zhu Hua, and Huang Jiangyong.

    Additionally, there was discussion on whether the nine-eight commission and large vessel taxes in Singapore should be restructured, given the current situation. It was agreed that further deliberation was necessary and that the matter should be referred back to the organizing committee for further consideration and a subsequent meeting to finalize the decision.
    Further, the issue of electing honorary members and revising the current bylaws to accommodate these changes was also discussed.
    """

    entities = NERPipeline.predict(
        sentence=sentence,
        tracer=tracer
    )

    print(entities)

    tracer.end()
    ArchSmith.persist()