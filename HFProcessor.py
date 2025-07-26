import os, threading
from services import Universal, FileOps
from addons import ASReport, ASTracer
from captioner import ImageCaptioning
from faceRecog import FaceRecognition, FaceDetection, FaceEmbedding
from fm import FileManager, File
from models import Figure, Face
from dotenv import load_dotenv
load_dotenv()

class HFProcessor:
    faceMatchLock = threading.Lock()
    
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
    def identifyFaces(imagePath: str, tracer: ASTracer) -> list[str]:
        imgName = os.path.basename(imagePath)
        
        face_crops = FaceRecognition.detect_face_crops(imagePath)
        if isinstance(face_crops, str):
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR IDENTIFYFACES ERROR",
                    message="Failed to extract face crops from image (fallback to empty list); response: {}".format(face_crops),
                    extraData={"imageName": imgName}
                )
            )
            return []
        
        tracer.addReport(
            ASReport(
                source="HFPROCESSOR IDENTIFYFACES",
                message="Extracted {} face crops from image '{}'.".format(len(face_crops), imgName)
            )
        )
        
        embed_values = None
        try:
            embed_values = [FaceRecognition.get_face_embedding(FaceRecognition.img_to_tensor(crop)) for crop in face_crops]
        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR IDENTIFYFACES ERROR",
                    message="Failed to get face embeddings (fallback to empty list); response: {}".format(e),
                    extraData={"imageName": imgName, "faceCropsLength": len(face_crops)}
                )
            )
            return []

        tracer.addReport(
            ASReport(
                source="HFPROCESSOR IDENTIFYFACES",
                message="Generated {} face embeddings for image '{}'.".format(len(embed_values), imgName),
                extraData={"faceCropsLength": len(face_crops)}
            )
        )
        
        matchedFigureIDs: set[str] = set()
        
        index = 0
        for embed in embed_values:
            with HFProcessor.faceMatchLock:
                # Carry out face matching
                figures: list[Figure] = []
                try:
                    figures = Figure.load(withEmbeddings=True)
                except Exception as e:
                    tracer.addReport(
                        ASReport(
                            source="HFPROCESSOR IDENTIFYFACES ERROR",
                            message="Failed to load figures to match face crop at index {} (skipping); error: {}".format(index, e),
                            extraData={"imageName": imgName}
                        )
                    )
                    continue
                
                embed = FaceEmbedding(value=embed, origin=imgName)
                matched = False
                for figure in figures:
                    # See if the embedding matches with this figure
                    try:
                        out = FaceDetection.match(
                            target=embed,
                            candidates=list(figure.face.embeddings.values()),
                            strategy=FaceDetection.MatchStrategy.VOTING,
                            threshold=0.7,
                            diversity_threshold=0.8
                        )
                        
                        if isinstance(out, str):
                            raise Exception(out)
                        
                        ## Extract the match result
                        matchResult, _ = out
                        if not isinstance(matchResult, FaceDetection.MatchResult):
                            raise ValueError("Invalid match result type: {}".format(type(matchResult)))
                    except Exception as e:
                        tracer.addReport(
                            ASReport(
                                source="HFPROCESSOR IDENTIFYFACES ERROR",
                                message="Failed to obtain embedding match result for figure '{}'; error: {}".format(figure.id, e),
                                extraData={"imageName": imgName, "faceCropIndex": index}
                            )
                        )
                        continue
                    
                    if matchResult != FaceDetection.MatchResult.NO_MATCH:
                        # If a match is found, we add the figure ID to the matched set
                        matchedFigureIDs.add(figure.id)
                        
                        if matchResult == FaceDetection.MatchResult.MATCH_DIVERSE:
                            # If the match is diverse, we add the embedding to the figure
                            try:
                                embedID = figure.face.addEmbedding(embed, autoSave=True)
                                
                                tracer.addReport(
                                    ASReport(
                                        source="HFPROCESSOR IDENTIFYFACES",
                                        message="Added new embedding to figure '{}' for face crop at index {}.".format(figure.id, index),
                                        extraData={"imageName": imgName, "faceCropIndex": index, 'embedID': embedID}
                                    )
                                )
                            except Exception as e:
                                tracer.addReport(
                                    ASReport(
                                        source="HFPROCESSOR IDENTIFYFACES ERROR",
                                        message="Failed to add diverse embedding to figure '{}'; error: {}".format(figure.id, e),
                                        extraData={"imageName": imgName, "faceCropIndex": index}
                                    )
                                )
                        
                        # If a match is found, we can stop checking further figures
                        matched = True
                        break
                
                if not matched:
                    # If no match found, create a new Figure and Face
                    try:
                        # Save to filesystem and then to FileManager context
                        label = Universal.generateUniqueID()
                        file = File('{}.jpg'.format(label), 'people')
                        
                        face_crops[index].save(file.path())
                        
                        res = FileManager.save(file=file)
                        if isinstance(res, str):
                            tracer.addReport(
                                ASReport(
                                    source="HFPROCESSOR IDENTIFYFACES ERROR",
                                    message="Failed to save face crop image (fallback to empty list); response: {}".format(res),
                                    extraData={"imageName": imgName, "faceCropIndex": index}
                                )
                            )
                            continue
                        
                        # Construct a new Figure and add the embedding and save to database
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
                                source="HFPROCESSOR IDENTIFYFACES",
                                message="Created new figure '{}' for unmatched face crop at index {}.".format(newFigure.id, index),
                                extraData={"imageName": imgName, "faceCropIndex": index}
                            )
                        )
                    except Exception as e:
                        tracer.addReport(
                            ASReport(
                                source="HFPROCESSOR IDENTIFYFACES ERROR",
                                message="Failed to create new figure for unmatched face crop at index {}; error: {}".format(index, e),
                                extraData={"imageName": imgName, "faceCropIndex": index}
                            )
                        )
            
            index += 1
        
        # Process the matchedFigureIDs result
        matchedFigureIDs = list(matchedFigureIDs)
        if len(matchedFigureIDs) == 0:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR IDENTIFYFACES WARNING",
                    message="No figures matched or created for image '{}'.".format(imgName)
                )
            )
        else:
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR IDENTIFYFACES",
                    message="Processed image '{}' with {} matched figures.".format(imgName, len(matchedFigureIDs)),
                    extraData={"matchedFigures": matchedFigureIDs}
                )
            )
        
        return matchedFigureIDs
    
    @staticmethod
    def process(imagePath: str, tracer: ASTracer, useLLMCaptionImprovement: bool=True) -> dict | str:
        if not FaceRecognition.initialised:
            HFProcessor.setup()
        
        figureIDs = HFProcessor.identifyFaces(imagePath, tracer)
        
        caption = HFProcessor.captioningProcess(imagePath, tracer, useLLMCaptionImprovement)
        if caption.startswith("ERROR"):
            tracer.addReport(
                ASReport(
                    source="HFPROCESSOR PROCESS ERROR",
                    message=f"Captioning failed for image '{imagePath}': {caption}"
                )
            )
            caption = None
        
        output = {
            "figureIDs": figureIDs,
            "caption": caption
        }
        
        tracer.addReport(
            ASReport(
                source="HFPROCESSOR PROCESS",
                message="Processed image '{}'.".format(imagePath),
                extraData=output
            )
        )
        
        return output