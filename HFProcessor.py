from services import Universal, FileOps
from addons import ASReport, ASTracer
from captioner import ImageCaptioning
from faceRecog import FaceRecognition, FaceDetection, FaceEmbedding
from fm import FileManager, File
from models import Figure, Face
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
                    message="Failed to generate caption for image '{}': {}".format(imagePath, e)
                )
            )
            return "ERROR: Failed to generate caption for image '{}': {}".format(imagePath, e)
    
    # Face Recognition + Detection and Matching
    @staticmethod
    def processImage(imagePath: str, tracer: ASTracer) -> list[str]:
        face_crops = FaceRecognition.detect_face_crops(imagePath)
        if isinstance(face_crops, str):
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESSIMAGE ERROR",
                    message="Failed to extract face crops from image (fallback to empty list); response: {}".format(face_crops),
                    extraData={"imagePath": imagePath}
                )
            )
            return []
        
        embed_values = None
        try:
            embed_values = [FaceRecognition.get_face_embedding(FaceRecognition.img_to_tensor(crop)) for crop in face_crops]
        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESSIMAGE ERROR",
                    message="Failed to get face embeddings (fallback to empty list); response: {}".format(e),
                    extraData={"imagePath": imagePath, "faceCropsLength": len(face_crops)}
                )
            )
            return []
        
        # Carry out face matching
        figures: list[Figure] = []
        try:
            figures = Figure.load(withEmbeddings=True)
        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESSIMAGE ERROR",
                    message="Failed to load figures for face matching (fallback to empty list); response: {}".format(e),
                    extraData={"imagePath": imagePath}
                )
            )
            return []
        
        matchedFigureIDs: set[str] = set()
        
        index = 0
        for embed in embed_values:
            embed = FaceEmbedding(value=embed, origin=imagePath)
            matched = False
            for figure in figures:
                matchResult, _ = FaceDetection.match(
                    target=embed,
                    candidates=list(figure.face.embeddings.values()),
                    strategy=FaceDetection.MatchStrategy.VOTING,
                    threshold=0.7,
                    diversity_threshold=0.8
                )
                
                if matchResult != FaceDetection.MatchResult.NO_MATCH:
                    matchedFigureIDs.add(figure.id)
                    
                    if matchResult == FaceDetection.MatchResult.MATCH_DIVERSE:
                        figure.face.addEmbedding(embed, autoSave=True)
                    matched = True
            
            if not matched:
                # If no match found, create a new Figure and Face
                label = Universal.generateUniqueID()
                file = File('{}.jpg'.format(label), 'people')
                
                face_crops[index].save(file.path())
                
                res = FileManager.save(file=file)
                if isinstance(res, str):
                    tracer.addReport(
                        ASReport(
                            source="HFPROCESSOR PROCESSIMAGE ERROR",
                            message="Failed to save face crop image (fallback to empty list); response: {}".format(res),
                            extraData={"imagePath": imagePath, "faceCropIndex": index}
                        )
                    )
                    continue
                
                newFigure = Figure(
                    label=label,
                    headshot=file.filename,
                    face=None,
                    id=label
                )
                newFigure.face.addEmbedding(embed, autoSave=True)
                newFigure.save()
                matchedFigureIDs.add(newFigure.id)
                
                tracer.addReport(
                    ASReport(
                        source="HFPROCESSOR PROCESSIMAGE",
                        message="Created new figure '{}' for unmatched face crop at index {}.".format(newFigure.id, index),
                        extraData={"imagePath": imagePath, "faceCropIndex": index}
                    )
                )
            
            index += 1
        
        matchedFigureIDs = list(matchedFigureIDs)
        if len(matchedFigureIDs) == 0:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESSIMAGE WARNING",
                    message="No figures matched or created for image '{}'.".format(imagePath)
                )
            )
        else:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESSIMAGE",
                    message="Processed image '{}' with {} matched figures.".format(imagePath, len(matchedFigureIDs)),
                    extraData={"matchedFigures": matchedFigureIDs}
                )
            )
        
        return matchedFigureIDs
    
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
        
        # tracer.addReport(
        #     ASReport(
        #         source="HFPROCESSOR PROCESS",
        #         message=f"Processed image '{imagePath}'.",
        #         extraData={
        #             "faces": len(filenames),
        #             "caption": caption if caption else "No caption generated."
        #         }
        #     )
        # )
        
        # return {
        #     "faceFiles": filenames,
        #     "caption": caption
        # }
    
    # @staticmethod
    # def compare(img1: str, img2: str) -> bool | str:
    #     output = FaceRecognition.compareImagesFirebase(img1, img2)
    #     if isinstance(output, str):
    #         return output
        
    #     result, similarity = output
    #     return result, similarity
