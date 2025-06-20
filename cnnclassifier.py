from email.mime import image
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
    model: CNNModel = None
    transform = None
    device = None
    class_names = ['cc', 'hf']
    modelPath = os.path.join("models", "SCCCIClassifier.pth")

    @staticmethod
    def load_model(model_weights=None, device=None):
        if device and not isinstance(device, torch.device):
            raise ValueError("Device must be a torch.device or None.")

        ImageClassifier.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else device
        )

        ImageClassifier.model = CNNModel()
        model_weights = model_weights or ImageClassifier.modelPath

        checkpoint = torch.load(model_weights, map_location=ImageClassifier.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        ImageClassifier.model.load_state_dict(state_dict, strict=False)

        ImageClassifier.model.to(ImageClassifier.device)
        ImageClassifier.model.eval()

        ImageClassifier.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return ImageClassifier

    @staticmethod
    def predict(input_path, tracer: ASTracer):
        """
        Performs inference on a single image file or directory of images.
        Returns a list of (path, predicted_label, confidence).
        """
        if isinstance(input_path, str):
            if os.path.isdir(input_path):
                image_paths = [os.path.join(input_path, f)
                               for f in os.listdir(input_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            else:
                image_paths = [input_path]
        elif isinstance(input_path, list):
            image_paths = input_path
        else:
            raise ValueError("Input must be a file path, directory path, or list of file paths")

        results = []

        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                image = ImageClassifier.transform(image).unsqueeze(0).to(ImageClassifier.device)

                with torch.no_grad():
                    output = ImageClassifier.model(image)
                    prob = torch.sigmoid(output).item()
                    label_idx = 1 if prob > 0.5 else 0
                    label = ImageClassifier.class_names[label_idx]
                    confidence = prob * 100 if label_idx == 1 else (1 - prob) * 100

                results.append((path, label, confidence))

                # Log report if tracer is provided

                if tracer:
                    report = ASReport(
                        source=os.path.basename(path),
                        message=f"Prediction: {label} ({confidence:.2f}%)",
                        extraData={"probability": prob, "label_index": label_idx}
                    )
                    tracer.addReport(report)

            except Exception as e:
                print(f"Error processing {path}: {e}")

        return results

# === Entry point for testing the classifier ===
if __name__ == "__main__":
    ModelStore.setup()

    ModelStore.registerLoadModelCallbacks(
        cnn=ImageClassifier.load_model
    )

    ModelStore.loadModels("cnn")

    cnn_ctx = ModelStore.getModel("cnn")

    # Sanity test
    test_tensor = torch.randn(1, 3, 224, 224).to(ImageClassifier.device)
    print(f"CNN model output shape: {ImageClassifier.model(test_tensor).shape}")

    # Predict on a sample image
    results = ImageClassifier.predict("Companydata/CC/Sample 1/-003.jpg", tracer=None)

    # Print results
    for path, label, confidence in results:
        print(f"{path}: {label} - {confidence:.1f}%")