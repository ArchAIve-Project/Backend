import os
from services import FileOps
from addons import ArchSmith, ASReport
from cnnclassifier import ImageClassifier
from MMProcessor import MMProcessor
from HFProcessor import HFProcessor

class MetadataGenerator:
    @staticmethod
    def generate(relativeImagePath: str, allowLLMForCorrection: bool=True, allowLLMForCaptionImprovement: bool=True, captioningEncoder: str='vit') -> dict | str:
        fullPath = os.path.join(os.getcwd(), relativeImagePath)
        
        if not FileOps.exists(fullPath, type="file"):
            return "ERROR: File does not exist."
        
        tracer = ArchSmith.newTracer("Metadata generation for '{}'".format(relativeImagePath))
        tracer.addReport(
            ASReport(
                source="METADATAGENERATOR GENERATE",
                message="Starting generation pipeline. Identifying type of image first with ImageClassifier.",
                extraData={"imagePath": fullPath}
            )
        )
        
        # Identify which pipeline to invoke with the CNN Classifier
        output = None
        predictions = ImageClassifier.predict(fullPath, tracer)
        if len(predictions) == 0:
            tracer.addReport(
                ASReport(
                    "METADATAGENERATOR GENERATE ERROR",
                    "No predictions returned from ImageClassifier. Cannot proceed.",
                    extraData={"response": predictions}
                )
            )
        else:
            prediction = predictions[0]
            imageType: str = prediction[1]
            if imageType not in ['cc', 'hf']:
                tracer.addReport(
                    ASReport(
                        "METADATAGENERATOR GENERATE ERROR",
                        "ImageClassifier returned an unsupported image type: {}. Cannot proceed.".format(imageType),
                        extraData={"response": predictions}
                    )
                )
                return "ERROR: Unsupported image type returned by ImageClassifier."
            
            tracer.addReport(
                ASReport(
                    "METADATAGENERATOR GENERATE",
                    "ImageClassifier returned image type: {}. Triggering {}.".format(imageType, "MMProcessor" if imageType == 'cc' else "HFProcessor"),
                )
            )

            if imageType == 'cc':
                output = MMProcessor.mmProcess(fullPath, tracer, useLLMCorrection=allowLLMForCorrection)
                
                tracer.addReport(
                    ASReport(
                        "METADATAGENERATOR GENERATE",
                        "MMProcessor returned output.",
                        extraData={"response": output}
                    )
                )
            else:
                output = HFProcessor.process(fullPath, tracer, useLLMCaptionImprovement=allowLLMForCaptionImprovement, captioningEncoder=captioningEncoder)
                
                tracer.addReport(
                    ASReport(
                        "METADATAGENERATOR GENERATE",
                        "HFProcessor returned output.",
                        extraData={"response": output}
                    )
                )
        
        tracer.end()
        ArchSmith.persist()
        return output if output else "ERROR: No valid output generated."