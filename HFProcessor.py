import os, torch, uuid
from torch.nn import functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from services import Universal
from addons import ASReport, ASTracer
from captioner import ImageCaptioning, Vocabulary
from dotenv import load_dotenv
load_dotenv()

DEVICE = torch.device("cpu")

class FaceEmbedding:
    def __init__(self, embedding: torch.Tensor, added: str=None):
        if added is None:
            added = Universal.utcNowString()
        
        self.embedding = embedding
        self.added = added
    
    def represent(self) -> dict:
        return {
            "embedding": self.embedding.tolist(),
            "added": self.added
        }
    
    def compareWith(self, other: 'FaceEmbedding') -> float:
        return FaceEmbedding.similarity(self, other)
    
    @staticmethod
    def similarity(embedding1: 'FaceEmbedding', embedding2: 'FaceEmbedding') -> float:
        if not isinstance(embedding1, FaceEmbedding) or not isinstance(embedding2, FaceEmbedding):
            raise Exception("FACEEMBEDDING SIMILARITY ERROR: Both inputs must be FaceEmbedding instances.")
        
        e1, e2 = F.normalize(embedding1.embedding), F.normalize(embedding2.embedding)
        return F.cosine_similarity(e1, e2).item()
    
    @staticmethod
    def from_dict(data: dict) -> 'FaceEmbedding':
        embedding = torch.tensor(data['embedding'])
        return FaceEmbedding(embedding, data.get('added'))

class FaceRecognition:
    mtcnn: MTCNN = None
    resnet: InceptionResnetV1 = None
    initialised = False

    @staticmethod
    def setup() -> None:
        FaceRecognition.mtcnn = MTCNN(keep_all=True, device=DEVICE)
        FaceRecognition.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        FaceRecognition.initialised = True
    
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
        img = Image.open(img_path).convert("RGB")
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
        if not FaceRecognition.initialised:
            HFProcessor.setup()
        
        # TODO: Redo face recognition part.
        # filenames = FaceRecognition.processImage(imagePath, tracer)
        
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
