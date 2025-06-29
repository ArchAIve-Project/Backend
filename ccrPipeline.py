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
DEVICE= torch.device("mps" if torch.backends.mps.is_available()
                            else "cuda" if torch.cuda.is_available()
                            else "cpu")

# === Model Definitions ===

class ChineseClassifier(nn.Module):
    """
    A ResNet50-based model for multi-class Chinese character classification.

    Given an input image, the model extracts features using a ResNet50 backbone,
    passes them through a projection and classification head, and outputs logits
    corresponding to character class indices.

    ## Usage:
    ```python    
    model = ChineseClassifier(embed_dim=256, num_classes=1200)
    ```
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
    A binary classifier for filtering valid characters from noise in image segmentation.

    The model is based on ResNet50 and outputs a single logit indicating validity.
    It supports optional feature extraction via `return_embedding`.

    ## Usage:
    ```python
    model = ChineseBinaryClassifier(embed_dim=512, unfreezeEncoder=True)
    ```
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
    Full pipeline for transcribing handwritten Chinese characters from scanned images.

    This class provides methods for image segmentation, character filtering,
    recognition, LLM-based correction, and accuracy evaluation.
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
    def checkFileType(image_path: str, tracer: ASTracer):
        # Extract extension, normalize
        ext = os.path.splitext(image_path)[1].lower().replace('.', '')
        if ext not in ['jpg', 'jpeg', 'jpe', 'png']:
            tracer.addReport(ASReport("CCRPIPELINE CHECKFILETYPE ERROR", f"File type of {ext} not supported."))
            return f"ERROR: Unsupported image file type: .{ext}"

        imageFileType = f"image/{'jpeg' if ext in ['jpg', 'jpeg', 'jpe'] else 'png'}"
        return imageFileType

    @staticmethod
    def load_label_mappings():
        """
        Loads and returns a mapping from class indices to Chinese characters.

        The mappings are read from 'ccr_labels.json' located in the model store directory.
        """
        labels_file = os.path.join(ModelStore.rootDirPath(), "ccr_labels.json")
        with open(labels_file, "r", encoding="utf-8") as f:
            label_dict = json.load(f)
        unique_chars = sorted(set(label_dict.values()))
        return {idx: char for idx, char in enumerate(unique_chars)}

    @staticmethod
    def loadChineseClassifier(modelPath: str):
        """
        Loads a trained Chinese character classifier from a checkpoint.

        Initializes the model, loads weights from the specified path, and prepares
        it for inference on the configured device.

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
        Loads a trained binary classifier for filtering valid characters from noise.

        Initializes the binary classifier model, loads weights from the given checkpoint,
        and sets it to evaluation mode on the configured device.
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
        Pads a grayscale image to a square shape by centering it and adding borders.

        The original image is centered on a square black background, then additional padding
        is added around all sides.

        Args:
            image (np.ndarray): Input 2D grayscale image array.
            padding (int, optional): Number of pixels to pad on each side after squaring. Defaults to 40.

        Returns:
            np.ndarray: The padded square image with extra border.

        ## Usage:
        ```python
        padded_img = CCRPipeline.padToSquare(img, padding=40)
        ```
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
        Segments a scanned handwritten Chinese document image into individual character images.

        The segmentation follows traditional right-to-left, top-to-bottom reading order.
        It performs grayscale conversion, binarization, morphological noise removal,
        column detection via vertical projection, and character row detection within columns.

        Each detected character image is padded to a square shape for uniformity.

        Args:
            image_path (str): Path to the input image file.
            tracer (ASTracer): Tracer object for logging segmentation reports.

        Returns:
            List[Tuple[int, np.ndarray]]: List of tuples containing (column_index, padded_character_image).

        ## Usage:
        ```python
        chars = CCRPipeline.segmentImage("docs/handwritten_sample.jpg", tracer)
        for col_idx, char_img in chars:
            # Process each character image
            pass
        ```
        """

        # Read the image in color
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            tracer.addReport(ASReport("CCRPIPELINE SEGMENTIMAGE ERROR", f"Could not read image: {image_path}"))
            return f"Could not read image: {image_path}"

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
        Filters character images to remove noise and recognizes valid characters.

        Uses a binary classifier to filter out noisy/non-character images,
        then applies the main recognition model to predict Chinese characters.
        Returns a dictionary mapping column indices to lists of recognized characters.

        Args:
            char_images (List[Tuple[int, np.ndarray]]): List of (column_index, character_image) tuples.
            recog_model (nn.Module): Multi-class Chinese character classification model.
            idx2char (Dict[int, str]): Mapping from class indices to Chinese characters.
            ccrCharFilter (nn.Module): Binary classifier model for filtering noise.
            tracer (ASTracer): Tracer object for logging recognition reports.

        Returns:
            Dict[int, List[str]]: Mapping from column index to recognized character list.

        ## Usage:
        ```python
        recognized_chars = CCRPipeline.detectCharacters(char_images, recog_model, idx2char, ccrCharFilter, tracer)
        ```
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
        Concatenates recognized characters into a final text string in reading order.

        Preserves the traditional right-to-left, top-to-bottom reading sequence
        by joining characters column-wise from left to right.

        Args:
            result_dict (Dict[int, List[str]]): Mapping of column indices to recognized characters.
            tracer (ASTracer): Tracer object for logging the concatenation process.

        Returns:
            str: The reconstructed text string combining all recognized characters.

        ## Usage:
        ```python
        final_text = CCRPipeline.concatCharacters(recognized_chars, tracer)
        ```
        """

        concatText = "\n".join("".join(result_dict[col]) for col in sorted(result_dict.keys()))
        textCount = sum(len(chars) for chars in result_dict.values())
        tracer.addReport(ASReport("CCRPIPELINE CONCATCHARACTERS", f"Reconstructed final text. Number of text: {textCount}"))
        return concatText

    @staticmethod
    def llmCorrection(image_path: str, predicted_text: str, tracer, imageFileType: str) -> str:
        """
        Uses a large language model (LLM) to correct predicted Chinese calligraphy text
        based on the corresponding image.

        Sends the image and predicted text to the LLM with instructions to fix errors,
        missing characters, and confusing parts, returning only corrected Chinese text.
        Logs success or failure to the provided tracer.

        Args:
            image_path (str): File path to the input calligraphy image.
            predicted_text (str): Initial OCR-predicted Chinese text.
            tracer: Tracer object for logging correction results.

        Returns:
            str: LLM-corrected Chinese text, or the original prediction if correction fails.

        ## Usage:
        ```python
        corrected_text = CCRPipeline.llmCorrection(image_path, predicted_text, tracer)
        ```
        """


        # Create a new interaction context for each call with the reused tool and callbacks.
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_VL_PLUS
        )

        # Add the user interaction to the context:
        # Ask the LLM to correct the predicted text based on the image.
        # The LLM is expected to only output corrected Chinese text.

        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=(
                    "Please review the image and correct the predicted Chinese text below. "
                    "Make sure the correction matches the characters in the image and makes sense in context. "
                    "Fix any errors, missing characters, or confusing parts. "
                    "Only output the corrected Chinese text—no explanations.\n\n"
                    f"Predicted text:\n{predicted_text.strip()}"
                ),
                imagePath=image_path,
                imageFileType=imageFileType
            ),
            imageMessageAcknowledged=True
        )

        try:
            response = LLMInterface.engage(cont)

            if isinstance(response, str):
                raise Exception("ERROR: Failed to carry out LLM correction; response: {}".format(response))

            corrected = response.content.strip()
            textCount = len(corrected)

            tracer.addReport(ASReport("CCRPIPELINE LLMCORRECTION", f"LLM correction applied. Final text count: {textCount}"))
            return corrected

        except Exception as e:
            tracer.addReport(ASReport("CCRPIPELINE LLMCORRECTION ERROR", f"LLM correction failed: {e}"))            
            return predicted_text  # fallback to uncorrected text

    @staticmethod
    def accuracy(pred_text: str, gtPath: str = None) -> float:
        """
        Calculates character-level accuracy between predicted and ground truth text
        using the Longest Common Subsequence (LCS) method.

        Args:
            pred_text (str): The predicted text string.
            gtPath (str): Explicit path to ground truth .txt file. Required.

        Returns:
            float: Accuracy score between 0 and 1.
        """

        if (not gtPath) or (not isinstance(gtPath, str)) or (not os.path.exists(gtPath)) or (not gtPath.lower().endswith('.txt')):
            return "Invalid ground truth file provided."

        with open(gtPath, 'r', encoding='utf-8') as f:
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

        # Traceback for match
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

        # Analyze mismatches
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

        match_count = len(matched_chars)
        acc = match_count / len(gt_text) if len(gt_text) > 0 else 0.0

        return acc

    @staticmethod
    def transcribe(image_path: str, tracer: ASTracer, computeAccuracy: bool = False, useLLMCorrection: bool = True, gtPath: str = None) -> dict:
        """
        Executes the full OCR transcription pipeline on a given image.

        Args:
            image_path (str): Path to the input image file.
            tracer (ASTracer): Tracer object for logging pipeline reports.
            computeAccuracy (bool, optional): If True, computes accuracy. Defaults to None.
            useLLMCorrection (bool, optional): If True, applies LLM correction. Defaults to True.
            gtPath (str, optional): Path to ground truth file. Required if computeAccuracy=True.

        Returns:
        ```json
        {
            "transcription": <final text>,
            "corrected": <bool>,
            "preCorrectionAccuracy": <float or None>,
            "postCorrectionAccuracy": <float or None>
        }
        ```
        """
        
        if computeAccuracy and not gtPath:
            return "ERROR: Ground truth file is required for accuracy computation."

        try:
            ccr_model_ctx = ModelStore.getModel("ccr")
            if not ccr_model_ctx or not ccr_model_ctx.model:
                return "ERROR: CCR model not loaded"

            ccr_filter_ctx = ModelStore.getModel("ccrCharFilter")
            if not ccr_filter_ctx or not ccr_filter_ctx.model:
                return "ERROR: CCR Char Filter model not loaded"

            if CCRPipeline.idx2char is None:
                CCRPipeline.idx2char = CCRPipeline.load_label_mappings()

            imageFileType = CCRPipeline.checkFileType(image_path, tracer)
            if imageFileType.startswith("ERROR"):
                return "ERROR: Unsupported file type provided. Please only provide .jpg .jpe .jpeg .png files."

            char_images = CCRPipeline.segmentImage(image_path, tracer)

            recognized = CCRPipeline.detectCharacters(
                char_images,
                recog_model=ccr_model_ctx.model,
                idx2char=CCRPipeline.idx2char,
                ccrCharFilter=ccr_filter_ctx.model,
                tracer=tracer
            )

            predText = CCRPipeline.concatCharacters(recognized, tracer)

            # Accuracy before correction
            preAcc = None
            postAcc = None

            if computeAccuracy:
                preAcc = CCRPipeline.accuracy(predText, gtPath)

            # LLM correction
            corrected = False
            if useLLMCorrection:
                try:
                    correctedText = CCRPipeline.llmCorrection(image_path, predText, tracer, imageFileType)
                    if correctedText != predText:
                        corrected = True
                except:
                    correctedText = predText
                    tracer.addReport(ASReport("CCRPIPELINE TRANSCRIBE ERROR", "Fallback to original predicted text due to LLM failure."))
            else:
                correctedText = predText
                tracer.addReport(ASReport("CCRPIPELINE TRANSCRIBE", "LLM correction skipped by user option."))

            if computeAccuracy and corrected:
                postAcc = CCRPipeline.accuracy(correctedText, gtPath)

            return {
                "transcription": correctedText,
                "corrected": corrected,
                "preCorrectionAccuracy": preAcc,
                "postCorrectionAccuracy": postAcc
            }

        except Exception as e:
            tracer.addReport(ASReport("CCRPIPELINE TRANSCRIBE ERROR", f"Error: {e}"))
            return f"ERROR: {e}"

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