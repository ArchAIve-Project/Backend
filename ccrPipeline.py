import os
import json
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import models, transforms
from addons import ModelStore, ASTracer, ASReport

# === Constants ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Model Definitions ===

class ChineseClassifier(nn.Module):
    """
    Multi-class character classification model using a ResNet50 backbone.
    Outputs a class index representing a Chinese character.
    """
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        base = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return self.classifier(x)


class ChineseBinaryClassifier(nn.Module):
    """
    Binary classifier used for filtering valid vs. noisy characters.
    Uses pretrained ResNet50 optionally.
    """
    def __init__(self, embed_dim: int = 512, unfreezeEncoder: bool = True):
        super().__init__()
        base = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(base.children())[:-1])

        # Allow optional fine-tuning of the encoder
        for param in self.resnet.parameters():
            param.requires_grad = unfreezeEncoder

        self.fc = nn.Linear(base.fc.in_features, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, return_embedding: bool = False):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x if return_embedding else self.classifier(x)


# === OCR Pipeline ===

class CCRPipeline:
    """
    The full pipeline to transcribe handwritten Chinese characters from a scanned image.
    """

    @staticmethod
    def load_label_mappings():
        """
        Loads label mappings from a JSON file stored in ModelStore.
        Converts class indices to Chinese characters.
        """
        labels_file = os.path.join(ModelStore.rootDirPath(), "ccr_labels.json")
        with open(labels_file, "r", encoding="utf-8") as f:
            label_dict = json.load(f)
        unique_chars = sorted(set(label_dict.values()))
        return {idx: char for idx, char in enumerate(unique_chars)}

    @staticmethod
    def loadModel(modelPath: str, modelName: str = None):
        """
        Loads either the main recognition model or the binary filter model
        based on the file name or explicitly provided name.
        """
        if modelName is None:
            modelName = "ccrCharFilter" if "ccrcharfilter" in modelPath.lower() else "ccr"

        if modelName == "ccrCharFilter":
            model = ChineseBinaryClassifier()
        else:
            model = ChineseClassifier(embed_dim=512, num_classes=1200)

        checkpoint = torch.load(modelPath, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model

    @staticmethod
    def padToSquare(image, tracer: ASTracer, padding: int = 40):
        """
        Adds padding around the character and centers it in a square image.
        Helps with consistent input shape for the classifier.
        """
        h, w = image.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = image
        tracer.addReport(ASReport("CCRPIPELINE PADTOSQUARE", "Image padded to square shape"))
        return cv2.copyMakeBorder(square, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    @staticmethod
    def segmentImage(image_path: str, tracer: ASTracer):
        """
        Segments a scanned handwritten document image into individual character images.
        Follows right-to-left, top-to-bottom reading order typical of traditional Chinese.
        """
        
        # Load image and convert to binary
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            error_msg = f"Could not read image: {image_path}"
            tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE ERROR", error_msg, {"error": True}))
            raise FileNotFoundError(error_msg)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Denoise and extract vertical column structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        vertical_proj = np.sum(denoised, axis=0)
        smooth_proj = cv2.GaussianBlur(vertical_proj.astype(np.float32), (25, 1), 0)
        threshold = np.max(smooth_proj) * 0.05
        in_column = smooth_proj > threshold

        # Detect columns (right to left)
        columns = []
        in_region = False
        for i, val in enumerate(in_column[::-1]):
            idx = len(in_column) - 1 - i
            if val and not in_region:
                end = idx
                in_region = True
            elif not val and in_region:
                start = idx
                if end - start >= 10:
                    columns.append((start, end))
                in_region = False
        if in_region:
            columns.append((0, end))

        # Segment characters in each column
        char_images = []
        for col_idx, (x1, x2) in enumerate(columns):
            col_img = denoised[:, x1:x2]
            if col_img.size == 0:
                continue
            col_img = cv2.resize(col_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            horizontal_proj = np.sum(col_img, axis=1)
            smooth_hproj = cv2.GaussianBlur(horizontal_proj.astype(np.float32), (15, 1), 0)
            threshold = np.max(smooth_hproj) * 0.05
            in_char = smooth_hproj > threshold

            char_regions = []
            in_region = False
            for i, val in enumerate(in_char):
                if val and not in_region:
                    top = i
                    in_region = True
                elif not val and in_region:
                    bottom = i
                    if bottom - top >= 20:
                        char_regions.append((top, bottom))
                    in_region = False
            if in_region:
                bottom = len(in_char) - 1
                if bottom - top >= 20:
                    char_regions.append((top, bottom))

            for char_idx, (y1, y2) in enumerate(char_regions):
                char_img = col_img[y1:y2, :]
                if char_img.size == 0:
                    continue
                padded_img = CCRPipeline.padToSquare(char_img, tracer)
                char_images.append((col_idx, padded_img))
                
        tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE", f"Segmented image: {image_path}"))

        return char_images

    @staticmethod
    def detectCharacters(char_images, recog_model, idx2char, ccrCharFilter, tracer: ASTracer):
        """
        Filters out noise using the binary classifier, and recognizes valid characters using the main model.
        Returns a dictionary mapping column index to character list.
        """
        
        transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        result = {}
        kept_images = []

        # First filter using binary classifier
        for col_idx, img in char_images:
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(ccrCharFilter(tensor)).item()
            if prob < 0.5:  # 0 = valid char
                kept_images.append((col_idx, img))

        # Recognize using multiclass classifier
        for col_idx, img in kept_images:
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = recog_model(tensor)
                pred_idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()
                result.setdefault(col_idx, []).append(idx2char[pred_idx])

        total = sum(len(chars) for chars in result.values())
        removed = len(char_images) - len(kept_images)
        tracer.addReport(ASReport("CCRPIPELINE DETECTCHARACTERS", f"{total} recognized, {removed} filtered out."))
        return result

    @staticmethod
    def concatCharacters(result_dict, tracer: ASTracer):
        """
        Combines the recognized characters into a final output string,
        preserving the RTL, top-down reading order.
        """
        concatText = "\n".join("".join(result_dict[col]) for col in sorted(result_dict.keys()))
        tracer.addReport(ASReport("CCRPIPELINE CONCATCHARACTERS", "Reconstructed final text."))
        return concatText

    @staticmethod
    def transcribe(image_path: str, tracer: ASTracer = None):
        """
        Full pipeline execution:
        - Load models from ModelStore
        - Segment image
        - Filter and recognize characters
        - Return final transcription
        """
        tracer = tracer or ASTracer("CCRPipeline_transcribe")
        tracer.start()

        try:
            ccr_model_ctx = ModelStore.getModel("ccr")
            if not ccr_model_ctx or not ccr_model_ctx.model:
                raise ValueError("CCR model not loaded")

            ccr_filter_ctx = ModelStore.getModel("ccrCharFilter")
            if not ccr_filter_ctx or not ccr_filter_ctx.model:
                raise ValueError("CCR Char Filter model not loaded")

            idx2char = CCRPipeline.load_label_mappings()
            char_images = CCRPipeline.segmentImage(image_path, tracer)

            recognized = CCRPipeline.detectCharacters(
                char_images,
                recog_model=ccr_model_ctx.model,
                idx2char=idx2char,
                ccrCharFilter=ccr_filter_ctx.model,
                tracer=tracer
            )
            
            finalText = CCRPipeline.concatCharacters(recognized, tracer)

            return finalText

        except Exception as e:
            tracer.addReport(ASReport("transcribe", f"Error: {e}", {"error": str(e)}))
            raise
        finally:
            tracer.end()


# === Entry Point ===

if __name__ == "__main__":
    # Setup the model registry
    ModelStore.setup()

    # Register loading callbacks for the models
    ModelStore.registerLoadModelCallbacks(
        ccr=CCRPipeline.loadModel,
        ccrCharFilter=CCRPipeline.loadModel
    )

    # Load both models
    ModelStore.loadModels("ccr")
    ModelStore.loadModels("ccrCharFilter")

    # Run simple model output tests
    test_tensor = torch.randn(1, 3, 224, 224)
    ccr_ctx = ModelStore.getModel("ccr")
    filter_ctx = ModelStore.getModel("ccrCharFilter")

    print(f"CCR model output shape: {ccr_ctx.model(test_tensor).shape}")
    print(f"Filter model output shape: {filter_ctx.model(test_tensor).shape}")
