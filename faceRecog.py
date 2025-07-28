import os, torch
from enum import Enum
from torch.nn import functional as F
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from services import Universal, FileOps

class FaceEmbedding:
    def __init__(self, value: torch.Tensor, origin: str=None, added: str=None):
        if added is None:
            added = Universal.utcNowString()
        
        self.value = value
        self.origin = origin
        self.added = added
    
    def represent(self) -> dict:
        return {
            "embedding": self.value.tolist(),
            "origin": self.origin,
            "added": self.added
        }
    
    def representForDB(self) -> dict:
        return {
            "origin": self.origin,
            "added": self.added
        }
    
    def compareWith(self, other: 'FaceEmbedding') -> float:
        return FaceEmbedding.similarity(self, other)
    
    def __str__(self):
        return "FaceEmbedding(value={}, origin='{}', added='{}')".format(
            self.value.shape if isinstance(self.value, torch.Tensor) else self.value,
            self.origin,
            self.added
        )
    
    @staticmethod
    def similarity(embedding1: 'FaceEmbedding', embedding2: 'FaceEmbedding') -> float:
        if not isinstance(embedding1, FaceEmbedding) or not isinstance(embedding2, FaceEmbedding):
            raise Exception("FACEEMBEDDING SIMILARITY ERROR: Both inputs must be FaceEmbedding instances.")
        
        e1, e2 = embedding1.value.to(Universal.getBestDevice(supported=['cuda', 'cpu'])), embedding2.value.to(Universal.getBestDevice(supported=['cuda', 'cpu']))
        
        e1, e2 = F.normalize(e1), F.normalize(e2)
        
        return F.cosine_similarity(e1, e2).item()
    
    @staticmethod
    def from_dict(data: dict) -> 'FaceEmbedding':
        embedding = torch.tensor(data['embedding'])
        return FaceEmbedding(embedding, data.get('origin'), data.get('added'))

class FaceRecognition:
    mtcnn: MTCNN = None
    resnet: InceptionResnetV1 = None
    initialised = False
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])
    device = Universal.getBestDevice(supported=['cuda', 'cpu'])

    @staticmethod
    def setup() -> None:
        FaceRecognition.mtcnn = MTCNN(keep_all=True, device=FaceRecognition.device)
        FaceRecognition.resnet = InceptionResnetV1(pretrained='vggface2', device=FaceRecognition.device).eval()
        FaceRecognition.initialised = True
    
    @staticmethod
    def detect_face_boxes(img_path) -> torch.Tensor | None | str:
        if not FileOps.exists(os.path.join(os.getcwd(), img_path), 'file'):
            return "ERROR: File does not exist: {}".format(img_path)
        
        try:
            img = Image.open(img_path)
        except Exception as e:
            return "ERROR: Failed to open image: {}".format(e)

        faces = FaceRecognition.mtcnn(img)
        if faces is None or len(faces) == 0:
            return None

        return faces

    @staticmethod
    def detect_face_crops(img_path, min_size: tuple[int, int]=(60, 60), min_conf=0.9) -> list[Image.Image] | str:
        if not FileOps.exists(os.path.join(os.getcwd(), img_path), 'file'):
            return "ERROR: File does not exist: {}".format(img_path)
        
        try:
            img = Image.open(img_path).convert("RGB")
            boxes, probs = FaceRecognition.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                return []
            
            cropped_faces = []
            for box, prob in zip(boxes, probs):
                if prob < min_conf:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                
                if width >= min_size[0] and height >= min_size[1]:
                    cropped_face = img.crop((x1, y1, x2, y2))
                    cropped_faces.append(cropped_face)
        except Exception as e:
            return "ERROR: Failed to detect face crops: {}".format(e)
        
        return cropped_faces
    
    @staticmethod
    def visualise_crop_tensors(cropped_faces: torch.Tensor):
        for face in cropped_faces:
            plt.imshow(face.permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.show()
    
    @staticmethod
    def draw_face_annotations(img_path, save_path=None) -> Image.Image | str:
        if not FileOps.exists(os.path.join(os.getcwd(), img_path), 'file'):
            return "ERROR: File does not exist: {}".format(img_path)
        
        img = Image.open(img_path)
        
        try:
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
        except Exception as e:
            return "ERROR: Failed to draw face annotations: {}".format(e)
        
        return img_draw
    
    @staticmethod
    def img_to_tensor(img: 'Image.Image | str', map_to: str=None) -> torch.Tensor | str:
        if isinstance(img, str):
            if not FileOps.exists(os.path.join(os.getcwd(), img), 'file'):
                return "ERROR: File does not exist: {}".format(img)
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            return "ERROR: Invalid image type. Must be PIL Image or file path."
        
        img: torch.Tensor = FaceRecognition.transform(img).to(map_to or FaceRecognition.device)
        
        return img
    
    @staticmethod
    def get_face_embedding(face_tensor: torch.Tensor) -> torch.Tensor | str:
        try:
            if face_tensor.ndim == 3:
                face_tensor = face_tensor.unsqueeze(0)
            
            face_tensor = face_tensor.to(torch.float32).to(FaceRecognition.device)
            
            embedding = None
            with torch.no_grad():
                embedding = FaceRecognition.resnet(face_tensor).detach().cpu()
        except Exception as e:
            return "ERROR: Failed to extract face embedding: {}".format(e)
        
        return embedding

class FaceDetection:
    class MatchStrategy(str, Enum):
        VOTING = "voting"
        ALL = "all"
        ANY = "any"
    
    class MatchResult(str, Enum):
        MATCH_DIVERSE = "match_diverse"
        MATCH = "match"
        NO_MATCH = "no_match"
    
    @staticmethod
    def match(target: FaceEmbedding, candidates: list[FaceEmbedding], strategy: MatchStrategy | str=MatchStrategy.VOTING, threshold: float=0.7, diversity_threshold: float=0.8) -> tuple[MatchResult, float] | str:
        try:
            strategy = FaceDetection.MatchStrategy(strategy) if isinstance(strategy, str) else strategy
            if not isinstance(strategy, FaceDetection.MatchStrategy):
                raise ValueError("Invalid 'strategy' provided.")
        except Exception as e:
            return "ERROR: Could not resolve match strategy; error: {}".format(e)
        
        result = FaceDetection.MatchResult.NO_MATCH
        scores = []
        
        try:
            scores = [FaceEmbedding.similarity(e, target) for e in candidates]
        except Exception as e:
            return "ERROR: Could not calculate similarity scores; error: {}".format(e)
        
        score = sum(scores) / len(scores) if scores else 0
        
        if strategy == FaceDetection.MatchStrategy.VOTING and sum(1 if s > threshold else 0 for s in scores) > (len(candidates) / 2):
            result = FaceDetection.MatchResult.MATCH
        elif strategy == FaceDetection.MatchStrategy.ALL and all(s > threshold for s in scores):
            result = FaceDetection.MatchResult.MATCH
        elif strategy == FaceDetection.MatchStrategy.ANY and any(s > threshold for s in scores):
            result = FaceDetection.MatchResult.MATCH
        
        if result == FaceDetection.MatchResult.MATCH:
            diverse_enough = True
            for s in scores:
                if s > diversity_threshold:
                    # Considered diverse if threshold < s < diversity_threshold
                    diverse_enough = False
                    break
            
            if diverse_enough:
                result = FaceDetection.MatchResult.MATCH_DIVERSE
        
        return result, score