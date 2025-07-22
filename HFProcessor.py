from addons import ASReport, ASTracer
from captioner import ImageCaptioning, Vocabulary
from faceRecog import FaceRecognition
from dotenv import load_dotenv
load_dotenv()

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
