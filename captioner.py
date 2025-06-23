import torch, os, pickle, re, time
from collections import Counter
from torch import nn
from torchvision import models, transforms
from PIL import Image
from addons import ModelStore, ArchSmith, ASTracer, ASReport

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {
            0: "pad",
            1: "startofseq",
            2: "endofseq",
            3: "unk"
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        self.index = 4
    
    def __len__(self):
        return len(self.itos)
    
    def tokenizer(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens
    
    def build_vocabulary(self, sentences):
        frequencies = Counter()
        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.index
                self.itos[self.index] = word
                self.index += 1
        
    def numericalize(self, text):
        tokens = self.tokenizer(text)
        numericalized = []
        for token in tokens:
            if token in self.stoi:
                numericalized.append(self.stoi[token])
            else:
                numericalized.append(self.stoi["unk"])
        return numericalized

class ResNetEncoderInfer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = True
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim, momentum=0.01)
 
    def forward(self, images):
        with torch.no_grad(): ## changed
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.batch_norm(features)
        return features

class DecoderLSTMInfer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, states):
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, states = self.lstm(inputs, states)
        logits = self.fc(lstm_out)
        return logits, states
 
    def generate(self, features, vocab_size, max_len=20): # changed
        batch_size = features.size(0)
        states = None
        generated_captions = []
 
        start_idx = 1  # startofseq
        end_idx = 2  # endofseq
        current_tokens = [start_idx]
 
        for _ in range(max_len):
            input_tokens = torch.LongTensor(current_tokens).to(features.device).unsqueeze(0)
            logits, states = self.forward(features, input_tokens, states)
            logits = logits.contiguous().view(-1, vocab_size)
            predicted = logits.argmax(dim=1)[-1].item()
 
            generated_captions.append(predicted)
            current_tokens.append(predicted)
 
        return generated_captions

class ImageCaptioningModelInference(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
 
    def generate(self, images, vocab_size, max_len): # changed
        features = self.encoder(images)
        return self.decoder.generate(features, vocab_size=vocab_size, max_len=max_len)

class ImageCaptioning:
    VOCAB_PATH = os.path.join(ModelStore.rootDirPath(), "vocab.pkl")
    MAX_SEQ_LENGTH = 25
    SEED = 42
    DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu")
    MIN_WORD_FREQ = 1
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    transform_inference = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    VOCAB = None
    
    @staticmethod
    def loadModel(modelPath: str):
        with open(ImageCaptioning.VOCAB_PATH, "rb") as f:
            ImageCaptioning.vocab = pickle.load(f)
        vocab_size = len(ImageCaptioning.vocab)

        encoder = ResNetEncoderInfer(embed_dim=ImageCaptioning.EMBED_DIM)
        decoder = DecoderLSTMInfer(ImageCaptioning.EMBED_DIM, ImageCaptioning.HIDDEN_DIM, vocab_size)
        model = ImageCaptioningModelInference(encoder, decoder).to(ImageCaptioning.DEVICE)

        state_dict = torch.load(modelPath, map_location=ImageCaptioning.DEVICE)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        
        return model
    
    @staticmethod
    def generateCaption(imagePath: str, tracer: ASTracer):
        ctx = ModelStore.getModel("imageCaptioner")
        if not ctx or not ctx.model:
            raise Exception("Model not found or not loaded in ModelStore.")
        
        try:
            ctxModel: ImageCaptioningModelInference = ctx.model
            pil_img = Image.open(imagePath).convert("RGB")
            img_tensor = ImageCaptioning.transform_inference(pil_img).unsqueeze(0).to(ImageCaptioning.DEVICE)

            startTime = time.time()
            with torch.no_grad():
                output_indices = ctxModel.generate(img_tensor, len(ImageCaptioning.vocab), max_len=ImageCaptioning.MAX_SEQ_LENGTH)
            inferenceTime = time.time() - startTime
        
            result_words = []
            end_token_idx = ImageCaptioning.vocab.stoi["endofseq"]
            for idx in output_indices:
                if idx == end_token_idx:
                    break
                word = ImageCaptioning.vocab.itos.get(idx, "unk")
                if word not in ["startofseq", "pad", "endofseq"]:
                    result_words.append(word)
            out = " ".join(result_words)
            
            tracer.addReport(
                ASReport(
                    source="IMAGECAPTIONING GENERATECAPTION",
                    message="Generated caption '{}' for image '{}'.".format(out, imagePath),
                    extraData={
                        "length": len(result_words),
                        "inferenceTime": "{:.4f}s".format(inferenceTime)
                    }
                )
            )
            
            return out
        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="IMAGECAPTIONING GENERATECAPTION ERROR",
                    message="Error generating caption for image '{}': {}".format(imagePath, str(e))
                )
            )
            return "ERROR: Failed to generate caption; error: {}".format(e)

if __name__ == "__main__":
    ModelStore.setup(imageCaptioner=ImageCaptioning.loadModel)
    
    # print(ModelStore.getModel("imageCaptioner").model)

    tracer = ArchSmith.newTracer("Test run of ImageCaptioning")
    
    print(ImageCaptioning.generateCaption("Companydata/63 20231117 ACCCIM hosted a forum with SCCCI.jpg", tracer))
    
    tracer.end()
    ArchSmith.persist()
    
    print('Done.')