import os, sys, json
from services import FileOps
from firebase import FireConn
from addons import ArchSmith, ASTracer, ASReport, ModelStore, ModelContext
from ai import LLMInterface
from fm import FileManager
from ccrPipeline import CCRPipeline
from cnnclassifier import ImageClassifier
from NERPipeline import NERPipeline
from captioner import ImageCaptioning, Vocabulary
from MMProcessor import MMProcessor
from HFProcessor import HFProcessor

class MetadataGenerator:
    @staticmethod
    def generate(relativeImagePath: str, allowLLMForCorrection: bool=True, allowLLMForCaptionImprovement: bool=True) -> dict | str:
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
                output = HFProcessor.process(fullPath, tracer, useLLMCaptionImprovement=allowLLMForCaptionImprovement)
                
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

if __name__ == "__main__":
    FireConn.connect()
    FileManager.setup()
    ModelStore.setup(
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier,
        ner=NERPipeline.load_model,
        cnn=ImageClassifier.load_model,
        imageCaptioner=ImageCaptioning.loadModel
    )
    LLMInterface.initDefaultClients()
    
    output = MetadataGenerator.generate("Companydata/-182.jpg")
    print(output)
    
    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print(f"Error: {e}")