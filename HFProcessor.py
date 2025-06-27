from addons import ModelStore, ArchSmith, ASTracer, ASReport
from captioner import ImageCaptioning, Vocabulary
from realEsrgan import RealESRGAN
import pickle

# HF Pipeline
class HFProcessor:
    """
    HFProcessor orchestrates a high-fidelity image processing pipeline.

    This class provides static methods that:
    - Enhance input images using a super-resolution model (Real-ESRGAN).
    - Generate captions for enhanced images using an image captioning model.

    Components:
    - enhanceImage(): Upscales the input image using Real-ESRGAN.
    - Captioning(): Generates a caption from an image using a pretrained model.
    - hfProcessor(): Full pipeline execution (enhancement + captioning).
    """
    # Image Enhancement (Using Real-ESRGAN)
    @staticmethod
    def enhanceImage(imagePath: str, outputPath: str, tracer: ASTracer):
        try:
            result = RealESRGAN().upscaleImage(imagePath, outputPath, tracer=tracer)

            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE ENHANCE IMAGE",
                    message=f"Enhanced image(s) saved to '{outputPath}'.",
                    extraData={"input": imagePath, "output": result}
                )
            )

            return result

        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE ENHANCE IMAGE ERROR",
                    message=f"Failed to enhance image: {e}",
                    extraData={"input": imagePath, "output": outputPath, "exception": str(e)}
                )
            )
            return f"ERROR: Failed to enhance image: {e}"


    # Image Captioning
    @staticmethod
    def Captioning(imagePath: str, tracer: ASTracer):
        """
        Generates a caption for the given image.
        Returns a string caption or an error message.
        """
        try:            
            # Ensure vocab is loaded
            if ImageCaptioning.vocab is None:
                with open(ImageCaptioning.VOCAB_PATH, "rb") as f:
                    ImageCaptioning.vocab = pickle.load(f)

            # Generate caption
            caption = ImageCaptioning().generateCaption(imagePath, tracer=tracer)

            # Log successful captioning
            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE CAPTIONING",
                    message=f"Captioning complete for '{imagePath}'.",
                    extraData={"caption": caption}
                )
            )

            return caption

        except Exception as e:
            # Log error in tracing
            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE CAPTIONING ERROR",
                    message=f"Failed to caption image '{imagePath}': {e}",
                    extraData={"input": imagePath, "exception": str(e)}
                )
            )
            return f"ERROR: Failed to caption image: {e}"
    
# Face Detection and Recognition
# Store in Metadata & Enriched Information

    # One that can call all of the above
    @staticmethod
    def hfProcessor(imagePath: str, tracer: ASTracer):
        """
        HF pipeline: enhance image and generate caption.
        Returns a dict with 'caption' and 'enhanced' keys or an error message string.
        """
        try:
            result = HFProcessor.enhanceImage(imagePath, "Companydata/Output", tracer)

            if isinstance(result, str) and result.startswith("ERROR"):
                return result

            enhancedPaths = result.get("successful", [])
            allResults = result.get("all_results", [])

            if not enhancedPaths:
                return "ERROR: No enhanced images generated."

            firstEnhanced = enhancedPaths[0]
            caption = HFProcessor.Captioning(firstEnhanced, tracer)

            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE PROCESSOR",
                    message="Full pipeline completed.",
                    extraData={
                        "input": imagePath,
                        "caption": caption,
                        "enhanced": enhancedPaths,
                        "details": allResults
                    }
                )
            )

            return {
                "caption": caption,
                "enhanced": enhancedPaths,
                "details": allResults
            }


        except Exception as e:
            tracer.addReport(
                ASReport(
                    source="HF IMAGE PIPELINE PROCESSOR ERROR",
                    message=f"Error during HFProcessor: {e}",
                    extraData={"input": imagePath, "exception": str(e)}
                )
            )
            return f"ERROR: Exception in HFProcessor: {e}"

if __name__ == "__main__":
    ModelStore.setup(
        imageCaptioner=ImageCaptioning.loadModel,
        realESRGAN=RealESRGAN.loadModel
    )
    
    tracer = ArchSmith.newTracer("HF Processor Test")

    imagePath = "Companydata/HF/02 Unidentified Photos (NYP)/1993-04-09 时任福州市委书记、市人大常委会主任、福州军分区党委第一书记习近平 (1).jpg"

    result = HFProcessor.hfProcessor(imagePath, tracer)

    if isinstance(result, dict):
        print(f"Enhanced Path(s): {result['enhanced']}")
        print(f"Caption: {result['caption']}")
    else:
        print(result)  # it's an error message

    tracer.end()
    ArchSmith.persist()