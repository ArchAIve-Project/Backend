import os
import torch
from torch import nn
from torch.nn import functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from addons import ArchSmith, ASReport, ASTracer, ModelStore
from firebase import FireConn
from fm import FileManager, File
from realEsrgan import RealESRGAN
from captioner import ImageCaptioning, Vocabulary
import pickle
from dotenv import load_dotenv
load_dotenv()

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu")

class FaceEmbedding:
    def __init__(self, embedding: torch.Tensor, img_path: str=None):
        self.embedding = embedding
        self.img_path = img_path
    
    def represent(self) -> dict:
        return {
            "embedding": self.embedding.tolist(),
            "img_path": self.img_path
        }
    
    @staticmethod
    def similarity(embedding1: 'FaceEmbedding', embedding2: 'FaceEmbedding') -> float:
        e1, e2 = F.normalize(embedding1.embedding), F.normalize(embedding2.embedding)
        return F.cosine_similarity(e1, e2).item()
    
    @staticmethod
    def from_dict(data: dict) -> 'FaceEmbedding':
        embedding = torch.tensor(data['embedding'])
        return FaceEmbedding(embedding, data.get('img_path'))

class FaceRecognition:
    mtcnn: MTCNN = None
    resnet: InceptionResnetV1 = None

    @staticmethod
    def setup() -> None:
        FaceRecognition.mtcnn = MTCNN(keep_all=True, device=DEVICE)
        FaceRecognition.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    
    @staticmethod
    def detect_face_boxes(img_path) -> torch.Tensor:
        if not os.path.exists(img_path):
            print(f"[ERROR] File does not exist: {img_path}")
            return None

        img = Image.open(img_path)
        faces = FaceRecognition.mtcnn(img)
        return faces

    @staticmethod
    def detect_face_crops(img_path) -> list[Image.Image] | None:
        img = Image.open(img_path)
        boxes, _ = FaceRecognition.mtcnn.detect(img)
        if boxes is None:
            return []
        
        cropped_faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img.crop((x1, y1, x2, y2))
            cropped_faces.append(cropped_face)
        
        return cropped_faces
    
    @staticmethod
    def visualise_crop_tensors(cropped_faces: torch.Tensor):
        for face in cropped_faces:
            plt.imshow(face.permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.show()
    
    @staticmethod
    def draw_face_annotations(img_path, save_path=None) -> Image.Image:
        img = Image.open(img_path)

        boxes, _, points = FaceRecognition.mtcnn.detect(img, landmarks=True)
        # Draw boxes and save faces
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        for _, (box, point) in enumerate(zip(boxes, points)):
            draw.rectangle(box.tolist(), width=5)
            for p in point:
                draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        
        if save_path:
            img_draw.save(save_path)
        
        return img_draw
    
    @staticmethod
    def get_face_embedding(face_tensor: torch.Tensor) -> torch.Tensor:
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        face_tensor = face_tensor.to(torch.float32).to(DEVICE)
        
        embedding = None
        with torch.no_grad():
            embedding = FaceRecognition.resnet(face_tensor).detach().cpu()
        
        return embedding

class HFProcessor:
    @staticmethod
    def setup():
        FaceRecognition.setup()
        FileManager.setup()
    
    @staticmethod
    def processImage(img_path: str, tracer: ASTracer):
        faceCrops = FaceRecognition.detect_face_crops(img_path)
        tracer.addReport(
            ASReport(
                source="FACEPROCESSOR PROCESSIMAGE",
                message="Detected {} faces in image '{}'.".format(len(faceCrops), img_path)
            )
        )
        
        face_index = 0
        for faceCrop in faceCrops:
            faceCrop.save(f"people/face_{face_index}.jpg")
            saveResult = FileManager.save("people", f"face_{face_index}.jpg")
            if isinstance(saveResult, str):
                tracer.addReport(ASReport("FACEPROCESSOR PROCESSIMAGE ERROR", f"Error saving face {face_index}: {saveResult}"))
            else:
                tracer.addReport(ASReport("FACEPROCESSOR PROCESSIMAGE", f"Face {face_index} saved successfully."))
            face_index += 1
        
        print("{} faces detected and saved.".format(face_index))
    
    # Compare with local file
    @staticmethod
    def compareImages(img_path1: str, img_path2: str, tracer= ASTracer) -> bool | str:
        """
        Compares two face images and checks if they belong to the same person.

        Args:
            img_path1 (str): Path to the first image.
            img_path2 (str): Path to the second image.

        Returns:
            bool | str: True if faces match, False if not, or an error string.
        """
        try:
            # Detect face tensors from both images
            faces1 = FaceRecognition.detect_face_boxes(img_path1)
            if faces1 is None or len(faces1) == 0:
                tracer.addReport(
                     ASReport(
                        source="FACEPROCESSOR COMPARE IMAGE 1 ERROR", 
                        message="ERROR: No face found in first image ({img_path1})"
                    )
                )

            faces2 = FaceRecognition.detect_face_boxes(img_path2)
            if faces2 is None or len(faces2) == 0:
                tracer.addReport(
                     ASReport(
                        source="FACEPROCESSOR COMPARE IMAGE 2 ERROR", 
                        message="ERROR: No face found in second image ({img_path2})"
                    )
                )

            # Get the first face embedding from each image
            emb1 = FaceRecognition.get_face_embedding(faces1[0])
            emb2 = FaceRecognition.get_face_embedding(faces2[0])

            sim = FaceEmbedding.similarity(FaceEmbedding(emb1), FaceEmbedding(emb2))
            result = sim > 0.6

            tracer.addReport(ASReport(
                source="FACEPROCESSOR COMPAREIMAGES",
                message=f"Compared faces from '{img_path1}' and '{img_path2}' — Match: {result}",
                extraData={"similarity": f"{sim:.4f}"}
            ))

            return result

        except Exception as e:
            tracer.addReport(ASReport(
                "FACEPROCESSOR COMPAREIMAGES EXCEPTION",
                "ERROR: Exception occurred while comparing images"))

    # Compare with firebase file
    @staticmethod
    def compareImagesFirebase(store: str, file1: str, file2: str):
        
        f1 = FileManager.prepFile(store, file1)
        f2 = FileManager.prepFile(store, file2)

        if isinstance(f1, str): return f1
        if isinstance(f2, str): return f2

        return HFProcessor.compareImages(f1.path(), f2.path())
    
    # Image Captioning
    @staticmethod
    def Captioning(image_path: str) -> str:
        """
        Generates a caption for the given image.
        Returns a string caption or an error message.
        """
        # Ensure model is loaded
        ctx = ModelStore.getModel("imageCaptioner")
        if ctx is None or ctx.model is None:
            raise RuntimeError("Image Captioning model not found or not loaded in ModelStore.")
        
        if ImageCaptioning.vocab is None:
            with open(ImageCaptioning.VOCAB_PATH, "rb") as f:
                ImageCaptioning.vocab = pickle.load(f)

        # Optional: create tracer if you want to record reports
        tracer = ASTracer("Image Captioning")

        return ImageCaptioning.generateCaption(image_path, tracer=tracer)

if __name__ == "__main__":
    FireConn.connect()

    ModelStore.setup(
        realESRGAN=RealESRGAN.loadModel,
        imageCaptioner=ImageCaptioning.loadModel
    )

    HFProcessor.setup()

    tracer = ArchSmith.newTracer("Test run of HF processor")
    tracer.addReport(
        ASReport(
            "TEST", 
            "Starting HF processor test run."
        )
    )
    
    inputImage = "Companydata/HF/02 Unidentified Photos (NYP)/51 20001020 Joint Secretary to Governemnt of India.jpg"

    RealESRGAN.upscaleImage(inputImage,"Companydata/Output",tracer=tracer)

    HFProcessor.processImage(inputImage, tracer)
    result = HFProcessor.compareImages("people/face_2.jpg", "people/face_3.jpg")
    result = HFProcessor.compareImagesFirebase("people","face_2.jpg", "face_3.jpg")
    print("Face Match Result:", result)

    caption = HFProcessor.Captioning(inputImage)
    print(f"Caption Result: {caption}")
    
    tracer.end()
    ArchSmith.persist()
    
    print("Done.")
