import os
import json
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import models, transforms
from addons import ModelStore, ASTracer, ASReport
from ai import LLMInterface, InteractionContext, Interaction, Tool, LMProvider, LMVariant

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
    
    idx2char = None    
    
    # Static transform reused for all image inputs
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

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
    def loadChineseClassifier(modelPath: str):
        """
        Standardized loader for the main Chinese character classifier model.
        """
        model = ChineseClassifier(embed_dim=512, num_classes=1200)
        checkpoint = torch.load(modelPath, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model

    @staticmethod
    def loadBinaryClassifier(modelPath: str):
        """
        Standardized loader for the binary character filter model.
        """
        model = ChineseBinaryClassifier()
        checkpoint = torch.load(modelPath, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        return model

    @staticmethod
    def padToSquare(image, padding: int = 40):
        """
        Adds padding around the character and centers it in a square image.
        """
        h, w = image.shape
        size = max(h, w)
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = image
        return cv2.copyMakeBorder(square, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    @staticmethod
    def segmentImage(image_path: str, tracer: ASTracer):
        """
        Segments a scanned handwritten document image into individual character images.
        Follows right-to-left, top-to-bottom reading order typical of traditional Chinese.
        """

        # Read the image in color
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            error_msg = f"Could not read image: {image_path}"
            tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE ERROR", error_msg))
            raise FileNotFoundError(error_msg)

        # Convert the image to grayscale and apply binarization (inverse + Otsu)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply morphological opening to clean up small noise (1x1 is minimal but useful)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Create a vertical projection profile to estimate column positions
        vertical_proj = np.sum(denoised, axis=0)
        smooth_proj = cv2.GaussianBlur(vertical_proj.astype(np.float32), (25, 1), 0)

        # Threshold the smoothed projection to find regions considered part of a column
        threshold = np.max(smooth_proj) * 0.05
        in_column = smooth_proj > threshold

        # Iterate the projection (right to left) to detect column boundaries
        columns = []
        in_region = False
        for i, val in enumerate(in_column[::-1]):
            idx = len(in_column) - 1 - i
            if val and not in_region:
                end = idx
                in_region = True
            elif not val and in_region:
                start = idx
                if end - start >= 10:  # Filter out narrow strips
                    columns.append((start, end))
                in_region = False
        if in_region:
            columns.append((0, end))  # Final column if still open

        # Process each column to detect characters within
        char_images = []
        padded_count = 0
        pad_errors = []

        for col_idx, (x1, x2) in enumerate(columns):
            col_img = denoised[:, x1:x2]
            if col_img.size == 0:
                continue

            # Scale up column for better resolution and separation
            col_img = cv2.resize(col_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

            # Generate horizontal projection profile to find characters within the column
            horizontal_proj = np.sum(col_img, axis=1)
            smooth_hproj = cv2.GaussianBlur(horizontal_proj.astype(np.float32), (15, 1), 0)

            # Threshold to identify horizontal spans where characters likely exist
            threshold = np.max(smooth_hproj) * 0.05
            in_char = smooth_hproj > threshold

            # Detect character row regions within the column
            char_regions = []
            in_region = False
            for i, val in enumerate(in_char):
                if val and not in_region:
                    top = i
                    in_region = True
                elif not val and in_region:
                    bottom = i
                    if bottom - top >= 20:  # Filter out tiny components
                        char_regions.append((top, bottom))
                    in_region = False
            if in_region:
                bottom = len(in_char) - 1
                if bottom - top >= 20:
                    char_regions.append((top, bottom))

            # Extract each character image and pad it to square
            for char_idx, (y1, y2) in enumerate(char_regions):
                char_img = col_img[y1:y2, :]
                if char_img.size == 0:
                    continue
                try:
                    padded_img = CCRPipeline.padToSquare(char_img)
                    char_images.append((col_idx, padded_img))
                    padded_count += 1
                except Exception as e:
                    pad_errors.append(str(e))

        # Report how many characters were padded and any errors that occurred
        summary_msg = f"{padded_count} images padded to square."
        if pad_errors:
            summary_msg += f" {len(pad_errors)} failed with errors: {', '.join(pad_errors[:3])}"
            if len(pad_errors) > 3:
                summary_msg += f", and {len(pad_errors) - 3} more."

        tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE PADTOSQUARE", summary_msg))
        tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE", f"Segmented image: {image_path}"))

        return char_images

    @staticmethod
    def detectCharacters(char_images, recog_model, idx2char, ccrCharFilter, tracer: ASTracer):
        """
        Filters out noise using the binary classifier, and recognizes valid characters using the main model.
        Returns a dictionary mapping column index to character list.
        """

        result = {}
        kept_images = []

        # First filter using binary classifier
        for col_idx, img in char_images:
            tensor = CCRPipeline.transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = torch.sigmoid(ccrCharFilter(tensor)).item()
            if prob < 0.5:  # 0 = valid char
                kept_images.append((col_idx, img))

        # Recognize using multiclass classifier
        for col_idx, img in kept_images:
            tensor = CCRPipeline.transform(img).unsqueeze(0).to(DEVICE)
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
        textCount = sum(len(chars) for chars in result_dict.values())
        tracer.addReport(ASReport("CCRPIPELINE CONCATCHARACTERS", f"Reconstructed final text. Number of text: {textCount}"))
        return concatText
    
    # Define the callback function for the LLM tool.
    def getccrCorrected(filePath: str, predText: str) -> str:
        """
        Simple passthrough callback for the tool that trims whitespace from the predicted text.
        """
        return predText.strip()

    # Pre-tool invocation callback to log or handle messages before tool runs.
    def preToolCallback(msg):
        print(f"\n[PRE-LLM] {msg}\n")

    # Post-tool invocation callback to log or handle messages after tool runs.
    def postToolCallback(msg):
        print(f"\n[POST-LLM] {msg}\n")

    # Define the LLM Tool once and reuse it for all calls.
    # This tool describes what parameters it expects and what it does.
    TOOL_GET_CCR_CORRECTED = Tool(
        callback=getccrCorrected,
        name="get_ccrCorrected",
        description="This tool can correct predicted Chinese Calligraphy text.",
        parameters=[
            Tool.Parameter(
                name="filePath",
                dataType=Tool.Parameter.Type.STRING,
                description="Image of Chinese Calligraphy meeting minutes.",
                required=True
            ),
            Tool.Parameter(
                name="predText",
                dataType=Tool.Parameter.Type.STRING,
                description="Predicted text from OCR or recognition model.",
                required=True
            )
        ]
    )
    
    @staticmethod
    def llmCorrection(image_path: str, predicted_text: str, tracer) -> str:
        """
        Sends the predicted Chinese calligraphy text and its corresponding image to an LLM for correction.
        Returns the LLM-corrected version of the text and logs the result to the tracer.
        """

        # Create a new interaction context for each call with the reused tool and callbacks.
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_VL_PLUS,
            tools=[CCRPipeline.TOOL_GET_CCR_CORRECTED],
            preToolInvocationCallback=CCRPipeline.preToolCallback,
            postToolInvocationCallback=CCRPipeline.postToolCallback
        )

        # Add the user interaction to the context:
        # Ask the LLM to correct the predicted text based on the image.
        # The LLM is expected to only output corrected Chinese text.
        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=f"Please review the image and correct the predicted Chinese text below. Make sure the correction matches the characters in the image and makes sense in context. Fix any errors, missing characters, or confusing parts. Only output the corrected Chinese text—no explanations.\n\nPredicted text:\n{predicted_text.strip()}",
                imagePath=image_path,
                imageFileType="image/jpg"
            ),
            imageMessageAcknowledged=True
        )

        try:
            response = LLMInterface.engage(cont)
            corrected = response.content.strip()
            textCount = len(corrected)
            
            tracer.addReport(ASReport("CCRPIPELINE LLMCORRECTION", f"LLM correction applied. Final text count: {textCount}"))
            return corrected
        except Exception as e:
            tracer.addReport(ASReport("CCRPIPELINE LLMCORRECTION ERROR", f"LLM correction failed: {e}"))
            return predicted_text  # fallback to uncorrected text
        
    @staticmethod
    def accuracy(pred_text: str, gt: str, show: bool = True) -> float:
        """
        Computes character-level accuracy using Longest Common Subsequence (LCS).

        - LCS finds the longest sequence of matching characters in order (not positionally).
        - Accuracy = matched characters (LCS) / total characters in ground truth.
        - More tolerant to extra, missing, or misaligned characters than strict comparison.
        - If show=True, prints matched, missing, extra characters, and basic stats.

        Returns:
            Accuracy as a float between 0 and 1.
        """
        
        # Read ground truth
        if not os.path.exists(gt):
            if show:
                print(f"[ERROR] Ground truth file '{gt}' not found.")
            return 0.0

        with open(gt, 'r', encoding='utf-8') as f:
            gt_text = f.read().strip()

        pred_text = pred_text.strip()

        # === LCS Computation ===
        m, n = len(gt_text), len(pred_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                if gt_text[i] == pred_text[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

        # Trace back to find the matched sequence
        i, j = m, n
        matched = []
        while i > 0 and j > 0:
            if gt_text[i - 1] == pred_text[j - 1]:
                matched.append(gt_text[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        matched.reverse()

        # Calculate missing and extra characters
        i = j = 0
        matched_chars = matched
        missing = []
        extra = []

        for c in matched_chars:
            while i < len(gt_text) and gt_text[i] != c:
                missing.append(gt_text[i])
                i += 1
            while j < len(pred_text) and pred_text[j] != c:
                extra.append(pred_text[j])
                j += 1
            i += 1
            j += 1

        missing.extend(gt_text[i:])
        extra.extend(pred_text[j:])

        # Accuracy
        match_count = len(matched_chars)
        acc = match_count / len(gt_text) if len(gt_text) > 0 else 0.0

        # === PRINT REPORT IF show ===
        if show:
            print("\n[Accuracy Report]")
            print(f"Ground truth length : {len(gt_text)}")
            print(f"Prediction length   : {len(pred_text)}")
            print(f"Matched characters  : {match_count}")
            print(f"Accuracy (LCS)      : {acc:.2%}")
            print("\nMatched sequence    :")
            print(f"{''.join(matched_chars)}")
            print("\nMissing (in GT)     :")
            print(f"{''.join(missing)}")
            print("\nExtra (in Pred)     :")
            print(f"{''.join(extra)}\n")

        return acc

    @staticmethod
    def transcribe(image_path: str, tracer: ASTracer):
        """
        Full pipeline execution:
        - Load models from ModelStore
        - Segment image
        - Filter and recognize characters
        - Corrects predicted transcription using an LLM
        - Returns final transcription
        """

        try:
            ccr_model_ctx = ModelStore.getModel("ccr")
            if not ccr_model_ctx or not ccr_model_ctx.model:
                raise ValueError("CCR model not loaded")

            ccr_filter_ctx = ModelStore.getModel("ccrCharFilter")
            if not ccr_filter_ctx or not ccr_filter_ctx.model:
                raise ValueError("CCR Char Filter model not loaded")

            # Load and cache idx2char only once
            if CCRPipeline.idx2char is None:
                CCRPipeline.idx2char = CCRPipeline.load_label_mappings()

            char_images = CCRPipeline.segmentImage(image_path, tracer)

            recognized = CCRPipeline.detectCharacters(
                char_images,
                recog_model=ccr_model_ctx.model,
                idx2char=CCRPipeline.idx2char,
                ccrCharFilter=ccr_filter_ctx.model,
                tracer=tracer
            )

            predText = CCRPipeline.concatCharacters(recognized, tracer)

            correctedText = CCRPipeline.llmCorrection(image_path, predText, tracer)
            
            # accuracy = CCRPipeline.accuracy(predText, "./ccr img/-064GT.txt", show=True)
            accuracy = CCRPipeline.accuracy(correctedText, "./ccr img/-064GT.txt", show=True)
            print(accuracy)

            tracer.end()
            # return predText
            return correctedText

        except Exception as e:
            tracer.addReport(ASReport("transcribe", f"Error: {e}", {"error": str(e)}))
            tracer.end()
            return str(e)

# === Entry Point ===

if __name__ == "__main__":
    # Setup the model registry
    ModelStore.setup()
    
    # Initialize the LLM client
    LLMInterface.initDefaultClients()

    # Register loading callbacks for the models
    ModelStore.registerLoadModelCallbacks(
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier
    )

    # Load both models
    ModelStore.loadModels("ccr", "ccrCharFilter")

    # Run simple model output tests
    test_tensor = torch.randn(1, 3, 224, 224)
    ccr_ctx = ModelStore.getModel("ccr")
    filter_ctx = ModelStore.getModel("ccrCharFilter")

    print(f"CCR model output shape: {ccr_ctx.model(test_tensor).shape}")
    print(f"Filter model output shape: {filter_ctx.model(test_tensor).shape}")