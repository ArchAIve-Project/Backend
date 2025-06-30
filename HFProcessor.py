import os
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from addons import ArchSmith, ASReport, ASTracer, ModelStore
from ai import LLMInterface
from firebase import FireConn
from fm import FileManager
# from realEsrgan import RealESRGAN
from captioner import ImageCaptioning, Vocabulary
import uuid
from dotenv import load_dotenv
load_dotenv()

DEVICE = torch.device("cpu")

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
    def detect_face_boxes(img_path) -> torch.Tensor | None | str:
        if not os.path.exists(img_path):
            return f"ERROR: File does not exist: {img_path}"

        try:
            img = Image.open(img_path)
        except Exception as e:
            return f"ERROR: Failed to open image: {e}"

        faces = FaceRecognition.mtcnn(img)
        if faces is None or len(faces) == 0:
            return None

        return faces

    @staticmethod
    def detect_face_crops(img_path) -> list[Image.Image] | None:
        img = Image.open(img_path)
        boxes, _ = FaceRecognition.mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        
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
    
    @staticmethod
    def processImage(img_path: str, tracer: ASTracer) -> list[str]:
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
        
        filenames = []
        for faceCrop in faceCrops:
            try:
                # Save to Local
                local = False
                cloud = False
                try:
                    uniqueId = uuid.uuid4().hex
                    faceCrop.save(f"people/face_{uniqueId}.jpg")
                    local = True
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
                        raise Exception(f"Unexpected response from FileManager: {saveResult}")
                    
                    cloud = True
                except Exception as e:
                    tracer.addReport(ASReport(
                        source="FACEPROCESSOR PROCESSIMAGE ERROR",
                        message=f"Exception saving face {uniqueId} to Firebase: {e}"
                    ))

                tracer.addReport(
                    ASReport(
                        source="FACEPROCESSOR PROCESSIMAGE",
                        message=f"Face {uniqueId} processing complete. See extra data for save status.",
                        extraData={"local": local, "cloud": cloud}
                    )
                )
                
                filenames.append(f"face_{uniqueId}.jpg")
            except Exception as e:
                tracer.addReport(ASReport(
                    source="FACEPROCESSOR PROCESSIMAGE ERROR",
                    message=f"Exception while processing face {uniqueId}: {e}"
                ))

        # print("{} faces detected and saved.".format(len(filenames)))
        return filenames

    # Compare with local file
    @staticmethod
    def compareImages(imgPath1: str, imgPath2: str) -> tuple | str:
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
            face1 = Image.open(imgPath1).convert("RGB")
            face1 = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])(face1)

            face2 = Image.open(imgPath2).convert("RGB")
            face2 = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])(face2)

            # Get the first face embedding from each image
            emb1 = FaceRecognition.get_face_embedding(face1)
            emb2 = FaceRecognition.get_face_embedding(face2)

            sim = FaceEmbedding.similarity(FaceEmbedding(emb1), FaceEmbedding(emb2))
            result = sim > 0.7

            return result, sim

        except Exception as e:
            return "ERROR: Exception occurred while comparing images; error: {}".format(e)

    # Compare with firebase file
    @staticmethod
    def compareImagesFirebase(file1: str, file2: str, store: str = "people"):

        f1 = FileManager.prepFile(store, file1)
        f2 = FileManager.prepFile(store, file2)

        if isinstance(f1, str):
            return f1

        if isinstance(f2, str):
            return f2

        return FaceRecognition.compareImages(f1.path(), f2.path())

class HFProcessor:
    @staticmethod
    def setup():
        FaceRecognition.setup()
    
    # Image Captioning
    @staticmethod
    def captioningProcess(imagePath: str, tracer: ASTracer, useLLMCaptionImprovement: bool=True) -> str:
        """
        Generates a caption for the given image.
        Returns a string caption or an error message.
        """
        try:
            caption = ImageCaptioning.generateCaption(imagePath, tracer=tracer, useLLMImprovement=useLLMCaptionImprovement)
            if caption.startswith("ERROR"):
                raise Exception(caption)
            
            return caption
        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="IMAGECAPTIONING CAPTIONINGPROCESS ERROR",
                    message=f"Failed to generate caption for image '{imagePath}': {e}"
                )
            )
            return f"ERROR: Failed to generate caption for image '{imagePath}': {e}"
    
    @staticmethod
    def process(imagePath: str, tracer: ASTracer, useLLMCaptionImprovement: bool=True) -> dict | str:
        filenames = FaceRecognition.processImage(imagePath, tracer)
        
        caption = HFProcessor.captioningProcess(imagePath, tracer, useLLMCaptionImprovement)
        if caption.startswith("ERROR"):
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESS ERROR",
                    message=f"Captioning failed for image '{imagePath}': {caption}"
                )
            )
            caption = None
        
        tracer.addReport(
            ASReport(
                source="HFPROCESSOR PROCESS",
                message=f"Processed image '{imagePath}'.",
                extraData={
                    "faces": len(filenames),
                    "caption": caption if caption else "No caption generated."
                }
            )
        )
        
        return {
            "faceFiles": filenames,
            "caption": caption
        }
    
    @staticmethod
    def compare(img1: str, img2: str) -> bool | str:
        output = FaceRecognition.compareImagesFirebase(img1, img2)
        if isinstance(output, str):
            return output
        
        result, similarity = output
        return result, similarity

if __name__ == "__main__":
    FireConn.connect()
    FileManager.setup()
    LLMInterface.initDefaultClients()

    ModelStore.setup(
        imageCaptioner=ImageCaptioning.loadModel
    )

    HFProcessor.setup()
    
    input("breakpoint")
    print()
    tracer = ArchSmith.newTracer("HF Processor testing")
    output = HFProcessor.process("Companydata/44 19930428 Wang Daohan (4).jpg", tracer, useLLMCaptionImprovement=False)
    tracer.end()
    ArchSmith.persist()
    
    print("Output:", output)
    
    input("breakpoint")
    
    print()
    compare = input("Make a comparison? (y/n): ").lower() == "y"
    while compare:
        img1 = input("Enter image 1 for comparison: ")
        img2 = input("Enter image 2 for comparison: ")
        output = HFProcessor.compare(img1, img2)
        print("Comparison Result:", output)
        
        print()
        compare = input("Make another comparison? (y/n): ").lower() == "y"
    
    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print(f"Error: {e}")