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
import uuid
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
            print(f"File does not exist: {img_path}")
            return None

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Failed to open image: {e}")
            return None

        faces = FaceRecognition.mtcnn(img)
        if faces is None or len(faces) == 0:
            print(f"No faces detected in image: {img_path}")
            return None

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
        if not faceCrops:
            tracer.addReport(
                ASReport(
                    source="FACEPROCESSOR PROCESSIMAGE ERROR",
                    message=f"No faces detected in image '{img_path}'."
                )
            )
        else:
            tracer.addReport(
                ASReport(
                    source="FACEPROCESSOR PROCESSIMAGE",
                    message=f"Detected {len(faceCrops)} faces in image '{img_path}'."
                )
            )
        
        try:
            os.makedirs("people", exist_ok=True)
        except Exception as e:
             tracer.addReport(ASReport(
            source="FACEPROCESSOR 'PEOPLE' DIRECTORY CREATION ERROR",
            message=f"Failed to create 'people' directory: {e}"
        ))

        for faceCrop in faceCrops:
            try:
                # Save to Local
                try:
                    uniqueId = str(uuid.uuid4())
                    faceCrop.save(f"people/face_{uniqueId}.jpg")
                    tracer.addReport(ASReport(
                        source="FACEPROCESSOR PROCESSIMAGE",
                        message=f"Face {uniqueId} saved locally successfully."
                    ))
                except Exception as e:
                    tracer.addReport(ASReport(
                        source="FACEPROCESSOR PROCESSIMAGE ERROR",
                        message=f"Failed to save face {uniqueId} locally: {e}"
                    ))
                    continue
                
                # Save to Firebase
                try:
                    saveResult = FileManager.save("people", f"face_{uniqueId}.jpg")
                    if isinstance(saveResult, str):
                        tracer.addReport(ASReport(
                            source="FACEPROCESSOR PROCESSIMAGE ERROR",
                            message=f"Error saving face {uniqueId} to Firebase: {saveResult}"
                        ))
                    else:
                        tracer.addReport(ASReport(
                            source="FACEPROCESSOR PROCESSIMAGE",
                            message=f"Face {uniqueId} saved to Firebase successfully."
                        ))
                except Exception as e:
                    tracer.addReport(ASReport(
                        source="FACEPROCESSOR PROCESSIMAGE ERROR",
                        message=f"Exception saving face {uniqueId} to Firebase: {e}"
                    ))
                    
            except Exception as e:
                tracer.addReport(ASReport(
                source="FACEPROCESSOR PROCESSIMAGE ERROR",
                message=f"Exception while processing face {uniqueId}: {e}"
                ))
        
        print("{} faces detected and saved.".format(uniqueId))
    
    # Compare with local file
    @staticmethod
    def compareImages(imgPath1: str, imgPath2: str, tracer: ASTracer) -> bool | str:
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
            faces1 = FaceRecognition.detect_face_boxes(imgPath1)
            if faces1 is None or len(faces1) == 0:
                tracer.addReport(
                     ASReport(
                        source="FACEPROCESSOR COMPARE IMAGE 1 ERROR", 
                        message="ERROR: No face found in first image ({imgPath1})"
                    )
                )

            faces2 = FaceRecognition.detect_face_boxes(imgPath2)
            if faces2 is None or len(faces2) == 0:
                tracer.addReport(
                     ASReport(
                        source="FACEPROCESSOR COMPARE IMAGE 2 ERROR", 
                        message="ERROR: No face found in second image ({imgPath2})"
                    )
                )

            # Get the first face embedding from each image
            emb1 = FaceRecognition.get_face_embedding(faces1[0])
            emb2 = FaceRecognition.get_face_embedding(faces2[0])

            sim = FaceEmbedding.similarity(FaceEmbedding(emb1), FaceEmbedding(emb2))
            result = sim > 0.6

            tracer.addReport(ASReport(
                source="FACEPROCESSOR COMPAREIMAGES",
                message=f"Compared faces from '{imgPath1}' and '{imgPath2}' — Match: {result}",
                extraData={"similarity": f"{sim:.4f}"}
            ))

            return result

        except Exception as e:
            tracer.addReport(ASReport(
                "FACEPROCESSOR COMPAREIMAGES EXCEPTION",
                "ERROR: Exception occurred while comparing images"
            ))
            return "ERROR: Exception occurred while comparing images; {}".format(e)

    # Compare with firebase file
    @staticmethod
    def compareImagesFirebase(file1: str, file2: str, store: str = "people"):

        f1 = FileManager.prepFile(store, file1)
        f2 = FileManager.prepFile(store, file2)

        if isinstance(f1, str):
            tracer.addReport(
                ASReport(
                    source="FACEPROCESSOR COMPAREFIREBASE ERROR",
                    message=f"Failed to prep file1: {f1}"
                )
            )
            return f1

        if isinstance(f2, str):
            tracer.addReport(
                ASReport(
                    source="FACEPROCESSOR COMPAREFIREBASE ERROR",
                    message=f"Failed to prep file2: {f2}"
                )
            )
            return f2

        return HFProcessor.compareImages(f1.path(), f2.path(), tracer)
    
    # Image Captioning
    @staticmethod
    def captioningProcess(imagePath: str, tracer: ASTracer) -> str:
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

        tracer.addReport(
            ASReport(
            source="IMAGECAPTIONING CAPTIONINGPROCESS",
            message=f"Generating caption for image '{imagePath}'"
            )
        )

        return ImageCaptioning.generateCaption(imagePath, tracer=tracer)

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
    
    inputImage = "Companydata/HF/Tay Beng Chuan/2.5.95中国国际商会上海商会代表团.jpg"

    # RealESRGAN.upscaleImage(inputImage,"Companydata/Output",tracer=tracer)

    HFProcessor.processImage(inputImage, tracer)

    # result = HFProcessor.compareImages("people/face_2.jpg", "people/face_3.jpg") 
    result = HFProcessor.compareImagesFirebase("face_6f1aa70b-aa58-4a09-b0f9-6527b0684aba.jpg", "face_cd7d61e1-5b6c-416d-912d-510ed9a87d50.jpg")
    print("Face Match Result:", result)

    caption = HFProcessor.captioningProcess(inputImage)
    print(f"Caption Result: {caption}")
    
    tracer.end()
    ArchSmith.persist()
    
    print("Done.")
