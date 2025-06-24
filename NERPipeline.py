import torch
from transformers import (
    BertTokenizerFast, BertConfig,
    BertForTokenClassification
)
from addons import ModelStore, ASTracer, ASReport

'''
A Named Entity Recognition (NER) pipeline using BERT for token classification.
This pipeline includes a tokenizer manager, label manager, model loader, and a predictor.
'''
# ======== Tokenizer Manager ========
class TokenizerManager:
    '''
    Manages the BERT tokenizer for token classification tasks.
    This class initializes the tokenizer from a pre-trained model. 
    '''
    def __init__(self, model_name_or_path: str):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)

    def get_tokenizer(self):
        return self.tokenizer

# ======== Label Manager ========
class DataHandler:
    '''
    Manages the labels for the NER task.
    This class provides methods to set and get label mappings.
    '''
    def __init__(self):
        self.label2id = {}
        self.id2label = {}

    def set_labels(self, label2id: dict, id2label: dict):
        self.label2id = label2id
        self.id2label = id2label

    def get_labels(self):
        return self.label2id, self.id2label

# ======== Model Loader ========
class ModelManager:
    '''
    Loads the BERT model for token classification.
    This class initializes the model with a configuration and loads the state dictionary from a specified path.
    '''
    def __init__(self, model_name: str, num_labels: int, id2label: dict, label2id: dict, pth_path: str):
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.model = BertForTokenClassification(config)
        state_dict = torch.load(pth_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict.get("model_state_dict", state_dict))
        self.model.eval()
        print(f"✅ Model loaded from weights: {pth_path}")

    def get_model(self):
        return self.model

# ======== Predictor ========
class NERPredictor:
    '''
    Predicts named entities in a sentence using a pre-trained BERT model. 
    This class takes a sentence, tokenizes it, and predicts the entity labels for each word in the sentence.
    '''
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label

    def predict(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)
        word_ids = inputs.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        # Align token predictions with word IDs
        aligned_preds = []
        previous_word_id = None
        for pred, word_id in zip(predictions, word_ids):
            if word_id is None or word_id == previous_word_id:
                continue
            aligned_preds.append(self.id2label[pred])
            previous_word_id = word_id

        words = sentence.split()
        return list(zip(words, aligned_preds))

# ======== NER Pipeline Wrapper ========
class NERPipeline:
    '''
    A pipeline for Named Entity Recognition (NER) using BERT.
    This class encapsulates the entire NER process, including model loading, tokenization, and prediction.
    It provides a simple interface to predict named entities in a given sentence.
    '''
    LABEL_LIST = [
        'B-ACT', 'B-DATE', 'B-LOC', 'B-MISC', 'B-ORG', 'B-PERSON',
        'I-ACT', 'I-DATE', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PERSON', 'O'
    ]
    MODEL_NAME = "bert-base-cased"

    def __init__(self, predictor: NERPredictor):
        self.predictor = predictor

    def predict(self, sentence: str):
        return self.predictor.predict(sentence)

    @staticmethod
    def load_model(model_path: str):
        label2id = {label: i for i, label in enumerate(NERPipeline.LABEL_LIST)}
        id2label = {i: label for i, label in enumerate(NERPipeline.LABEL_LIST)}

        tokenizer_mgr = TokenizerManager(NERPipeline.MODEL_NAME)
        tokenizer = tokenizer_mgr.get_tokenizer()

        model_mgr = ModelManager(
            model_name=NERPipeline.MODEL_NAME,
            num_labels=len(NERPipeline.LABEL_LIST),
            label2id=label2id,
            id2label=id2label,
            pth_path=model_path
        )
        model = model_mgr.get_model()

        predictor = NERPredictor(model, tokenizer, id2label)
        return NERPipeline(predictor)

# ======== Main Entrypoint ========
if __name__ == "__main__":
    ModelStore.setup(useDefaults=True)

    ModelStore.registerLoadModelCallbacks(
        ner=NERPipeline.load_model
    )
    ModelStore.loadModels("ner")

    model_ctx = ModelStore.getModel("ner")
    if model_ctx is None or model_ctx.model is None:
        raise RuntimeError("NER model not loaded")

    pipeline = model_ctx.model  # Already a NERPipeline instance
    sentence = "Qin Shi Huang was the first emperor of a unified China."
    predictions = pipeline.predict(sentence)

    for word, label in predictions:
        print(f"{word}: {label}")
