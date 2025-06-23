import json
import torch
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from modelstore import ModelStore

class TokenizerManager:
    def __init__(self, model_name_or_path):
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)

    def tokenize(self, text):
        return self.tokenizer(text, truncation=True, is_split_into_words=True)

    def get_tokenizer(self):
        return self.tokenizer


class DataHandler:
    def __init__(self, tokenizer: TokenizerManager):
        self.tokenizer = tokenizer
        self.label2id = {}
        self.id2label = {}

    def load_labels_from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            self.label2id = json.load(f)
        self.label2id = {str(k): int(v) for k, v in self.label2id.items()}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def set_labels(self, label2id: dict, id2label: dict):
        self.label2id = label2id
        self.id2label = id2label

    def get_labels(self):
        return self.label2id, self.id2label

    

class ModelManager:
    def __init__(self, model_name: str, num_labels: int, id2label: dict, label2id: dict, pth_path: str):
        # model config
        config = BertConfig.from_pretrained(
            model_name, 
            num_labels=num_labels, 
            id2label=id2label, 
            label2id=label2id
            )
        self.model = BertForTokenClassification.from_pretrained(model_name, config=config)
        self.model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {pth_path}")

    def get_model(self):
        return self.model
    

class NERPredictor:
    def __init__(self, model, tokenizer, id2label):
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label

    def predict(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)

        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        tokens = self.tokenizer.tokenize(sentence)

# Skip special tokens ([CLS], [SEP]) when aligning tokens and predictions
        aligned_preds = [self.id2label.get(str(pred), "O") for pred in predictions[1:len(tokens)+1]]

        return list(zip(tokens, aligned_preds))


# --- NER Pipeline ---
class NERPipeline:
    def __init__(self, model_name: str):
        self.tokenizer_mgr = TokenizerManager(model_name)
        self.label_mgr = DataHandler(self.tokenizer_mgr)
        self.model_mgr = None
        self.predictor = None
        self.model_name = model_name


    def load_labels(self, label2id_path: str, id2label_path: str):
        with open(label2id_path, 'r') as f:
            label2id = json.load(f)
        with open(id2label_path, 'r') as f:
            id2label = json.load(f)

        self.label_mgr.set_labels(label2id, id2label)  # ✅ Correct method call



    def load_model_from_pth(self, pth_path: str):
        label2id, id2label = self.label_mgr.get_labels()
        print("Loaded label2id:", label2id)
        print("Number of labels:", len(label2id))

        self.model_mgr = ModelManager(
            model_name=self.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            pth_path=pth_path
        )
        model = self.model_mgr.get_model()
        tokenizer = self.tokenizer_mgr.get_tokenizer()
        self.predictor = NERPredictor(model, tokenizer, id2label)

    def predict(self, sentence: str):
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model_from_pth() first.")
        return self.predictor.predict(sentence)


# --- Example usage ---
if __name__ == "__main__":
    ModelStore.setup()

    pipeline = NERPipeline("bert-base-cased")
    pipeline.load_labels("models\label2id.json", "models\id2label.json")
    pipeline.load_model_from_pth("bert_ner_weights.pth")

    sentence = "Qin Shi Huang was the first emperor of a unified China."
    predictions = pipeline.predict(sentence)
    print(predictions)