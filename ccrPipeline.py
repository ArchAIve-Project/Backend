from addons import ModelStore, ModelContext, ASTracer, ASReport
import torch
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch import nn
import os
import json

# Model Definition
class ChineseClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim)
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
    

class CCRPipeline:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_label_mappings():
        labels_file = os.path.join(ModelStore.rootDirPath(), "ccr_labels.json")
        with open(labels_file, "r", encoding="utf-8") as f:
            label_dict = json.load(f)
        all_labels = sorted(set(label_dict.values()))
        return {idx: char for idx, char in enumerate(all_labels)}

    @staticmethod
    def loadModel(modelPath: str):
        model = ChineseClassifier(embed_dim=512, num_classes=1200)
        checkpoint = torch.load(modelPath, map_location=CCRPipeline.DEVICE, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(CCRPipeline.DEVICE)
        model.eval()
        
        return model
    
    @staticmethod
    def padToSquare(image, tracer: ASTracer, padding=40):
        tracer.addReport(ASReport("padToSquare", "Padding image to square shape"))
        h, w = image.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = image
        padded = cv2.copyMakeBorder(square, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=0)
        return padded

    @staticmethod
    def segmentImage(image_path, tracer: ASTracer):
        tracer.addReport(ASReport("segmentImage", f"Segmenting image: {image_path}"))

        # Load image with error handling
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            error_msg = f"Could not read image: {image_path}"
            tracer.addReport(ASReport("segmentImage", error_msg, {"error": True}))
            raise FileNotFoundError(error_msg)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Denoising
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Vertical projection for columns
        vertical_proj = np.sum(denoised, axis=0)
        smooth_proj = cv2.GaussianBlur(vertical_proj.astype(np.float32), (25, 1), 0)
        threshold = np.max(smooth_proj) * 0.05
        in_column = smooth_proj > threshold

        # Find column regions from right to left (RTL)
        columns = []
        in_region = False
        for i, val in enumerate(in_column[::-1]):
            idx = len(in_column) - 1 - i
            if val and not in_region:
                end = idx
                in_region = True
            elif not val and in_region:
                start = idx
                if end - start >= 10:  # minimum column width
                    columns.append((start, end))
                in_region = False
        if in_region:
            columns.append((0, end))

        tracer.addReport(ASReport("segmentImage", f"Found {len(columns)} columns"))

        # Segment characters in each column
        char_images = []
        for col_idx, (x1, x2) in enumerate(columns):
            col_img = denoised[:, x1:x2]
            
            # Skip empty columns
            if col_img.size == 0:
                continue
                
            col_img = cv2.resize(col_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # Horizontal projection
            horizontal_proj = np.sum(col_img, axis=1)
            smooth_hproj = cv2.GaussianBlur(horizontal_proj.astype(np.float32), (15, 1), 0)
            char_threshold = np.max(smooth_hproj) * 0.05
            in_char = smooth_hproj > char_threshold

            # Detect character regions
            char_regions = []
            in_region = False
            for i, val in enumerate(in_char):
                if val and not in_region:
                    top = i
                    in_region = True
                elif not val and in_region:
                    bottom = i
                    if bottom - top >= 20:  # min height
                        char_regions.append((top, bottom))
                    in_region = False
            if in_region:
                bottom = len(in_char) - 1
                if bottom - top >= 20:
                    char_regions.append((top, bottom))

            tracer.addReport(ASReport("segmentImage", f"Column {col_idx}: Found {len(char_regions)} characters"))

            # Process characters
            for char_idx, (y1, y2) in enumerate(char_regions):
                char_img = col_img[y1:y2, :]
                
                # Skip empty character images
                if char_img.size == 0:
                    continue
                    
                padded_img = CCRPipeline.padToSquare(char_img, tracer)
                char_images.append((col_idx, padded_img))

        return char_images

    @staticmethod
    def detectCharacters(char_images, recog_model, idx2char, tracer: ASTracer):
        tracer.addReport(ASReport("detectCharacters", "Filtering & recognizing characters"))

        transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        result = {}
        for col_idx, img in char_images:
            input_tensor = transform(img).unsqueeze(0).to(CCRPipeline.DEVICE)

            with torch.no_grad():
                recog_output = recog_model(input_tensor)
                char_idx = torch.argmax(torch.softmax(recog_output, dim=1), dim=1).item()
                char = idx2char[char_idx]

                result.setdefault(col_idx, []).append(char)

        tracer.addReport(ASReport("detectCharacters", f"{sum(len(v) for v in result.values())} characters recognized"))
        return result

    @staticmethod
    def concatCharacters(result_dict, tracer: ASTracer):
        tracer.addReport(ASReport("concatCharacters", "Reconstructing final text (RTL)"))
        output_lines = []
        for col_idx in sorted(result_dict.keys()):
            output_lines.append("".join(result_dict[col_idx]))
        return "\n".join(output_lines)

    @staticmethod
    def transcribe(image_path, tracer: ASTracer = None):
        if tracer is None:
            tracer = ASTracer("CCRPipeline_transcribe")
        tracer.start()

        try:
            # Get the main model from ModelStore
            ccr_model_ctx = ModelStore.getModel("ccr")
            if not ccr_model_ctx or not ccr_model_ctx.model:
                raise ValueError("CCR model not loaded in ModelStore")

            # Load label mappings
            idx2char = CCRPipeline.load_label_mappings()

            # Process the image (without binary filtering)
            char_images = CCRPipeline.segmentImage(image_path, tracer)
            recognized = CCRPipeline.detectCharacters(
                char_images, 
                ccr_model_ctx.model,
                idx2char, 
                tracer
            )
            final_output = CCRPipeline.concatCharacters(recognized, tracer)

            tracer.addReport(ASReport("transcribe", "Transcription completed successfully"))
            return final_output
        
        except Exception as e:
            tracer.addReport(ASReport("transcribe", f"Error during transcription: {str(e)}", {"error": str(e)}))
            raise
        finally:
            tracer.end()
    

if __name__ == "__main__":
    ModelStore.setup()
    ModelStore.registerLoadModelCallbacks(ccr=CCRPipeline.loadModel)
    ModelStore.loadModels("ccr")

    model_ctx = ModelStore.getModel("ccr")
    test_tensor = torch.randn(1, 3, 224, 224)
    output = model_ctx.model(test_tensor)
    print(f"Output shape: {output.shape}")