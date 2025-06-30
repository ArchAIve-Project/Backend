import os, sys, json
from services import FileOps
from firebase import FireConn
from addons import ArchSmith, ASTracer, ASReport, ModelStore, ModelContext
from ai import LLMInterface
from ccrPipeline import CCRPipeline
from cnnclassifier import ImageClassifier
from NERPipeline import NERPipeline
from MMProcessor import MMProcessor

class MetadataGenerator:
    @staticmethod
    def generate(relativeImagePath: str):
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
            tracer.addReport(
                ASReport(
                    "METADATAGENERATOR GENERATE",
                    "ImageClassifier returned image type: {}".format(prediction[1])
                )
            )
            if prediction[1] == 'cc':
                output = MMProcessor.mmProcess(fullPath, tracer)
                
                tracer.addReport(
                    ASReport(
                        "METADATAGENERATOR GENERATE",
                        "MMProcessor returned data: {}".format(output)
                    )
                )
            else:
                output = "Human Figure Processor is a WIP."
        
        tracer.end()
        ArchSmith.persist()
        return output if output else "ERROR: No valid output generated."

if __name__ == "__main__":
    ModelStore.setup(
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier,
        ner=NERPipeline.load_model,
        cnn=ImageClassifier.load_model
    )
    LLMInterface.initDefaultClients()
    
    output = MetadataGenerator.generate("Companydata/-173.jpg")
    print(output)