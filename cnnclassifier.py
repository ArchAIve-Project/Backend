import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from addons import ModelStore, ASTracer, ASReport

"""
A CNN-based binary classifier for distinguishing between 'cc' and 'hf' classes.
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model Definition ===
class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_size=512, use_batchnorm=True):
        super(CNNModel, self).__init__()
        self.use_batchnorm = use_batchnorm
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(256 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# === Classifier Wrapper ===
class ImageClassifier:
    class_names = ["cc", "hf"]

    @staticmethod
    def load_model(modelPath: str):

        model = CNNModel() 

        checkpoint = torch.load(modelPath, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
        model.to(DEVICE)
        model.eval()

        ImageClassifier.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return model

    @staticmethod
    def predict(input_path, tracer: ASTracer):
        """
        Performs inference on a single image file or directory of images.
        Returns a list of (path, predicted_label, confidence).
        """

        model_ctx = ModelStore.getModel("cnn")
        if model_ctx is None or model_ctx.model is None:
            raise RuntimeError("CNN model is not loaded. Ensure it is registered and loaded in ModelStore.")
        model = model_ctx.model

        # if dir or file path is provided
        if isinstance(input_path, str):
            # dir
            if os.path.isdir(input_path):
                image_paths = [
                    os.path.join(input_path, f) 
                    for f in os.listdir(input_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                ]
            # file
            elif os.path.isfile(input_path):
                if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths = [input_path]
                else:
                    raise ValueError(f"File is not a supported image type: {input_path}")
            else:
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
        # list 
        elif isinstance(input_path, list):
            invalid_paths = [
                path for path in input_path
                if not isinstance(path, str) or
                not os.path.isfile(path) or
                not path.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            if invalid_paths:
                raise FileNotFoundError(
                    "Invalid or unsupported image files:\n" + "\n".join(invalid_paths)
                )

            image_paths = input_path

        else:
            raise ValueError("Input must be a file path, directory path, or list of file paths")

        results = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                image = ImageClassifier.transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = model(image)
                    prob = torch.sigmoid(output).item()
                    label_idx = 1 if prob > 0.5 else 0
                    label = ImageClassifier.class_names[label_idx]
                    confidence = prob * 100 if label_idx == 1 else (1 - prob) * 100

                results.append((path, label, confidence))

                # Log report if tracer is provided

                if tracer:
                    report = ASReport(
                        source="IMAGECLASSIFIER PREDICT",
                        message=f"Prediction: {label} ({confidence:.2f}%)",
                        extraData={"probability": prob, "label_index": label_idx}
                    )
                    tracer.addReport(report)

            except Exception as e:
                error_msg = f"Error processing {path}: {e}"
                print(error_msg)
                
                if tracer:
                    error_report = ASReport(
                        source="IMAGECLASSIFIER PREDICT ERROR",
                        message=error_msg,
                        extraData={"path": path, "exception": str(e)}
                    )
                    tracer.addReport(error_report)

        return results

# === Entry point for testing the classifier ===
if __name__ == "__main__":
    ModelStore.setup()

    ModelStore.registerLoadModelCallbacks(
        cnn=ImageClassifier.load_model
    
    )

    ModelStore.loadModels("cnn")

    model_ctx = ModelStore.getModel("cnn")
    if model_ctx is None or model_ctx.model is None:
        raise RuntimeError("CNN model is not loaded. Ensure it is registered and loaded in ModelStore.")
    model = model_ctx.model

    test_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    print(f"CNN model output shape: {model(test_tensor).shape}")

    # Predict on a sample image
    results = ImageClassifier.predict("Companydata/CC/Sample 1/-003.jpg", tracer=None)

    # Print results
    for path, label, confidence in results:
        print(f"{path}: {label} - {confidence:.1f}%")