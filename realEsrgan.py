import os
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from addons import ASTracer, ASReport, ArchSmith, ModelStore

class RealESRGAN:
    '''
    Real ESRGAN is a designed for restoring images with real-world degradations, 
    making it more practical for real-world applications compared to ESRGAN.
    '''
    DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu")

    @staticmethod
    def loadModel(modelPath: str):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=modelPath,
            model=model,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=RealESRGAN.DEVICE
        )

        return upsampler

    def upscaleImage(self, inputPath, outputPath: str, tracer: ASTracer):
        """
        Performs upscaling on single image file or directory of images.
        Returns an upscaled image in Output Folder with _upscaled of every image done.
        """

        # Check if model exist before image processing
        ctx = ModelStore.getModel("realESRGAN")
        if not ctx or not ctx.model:
            raise Exception("Model not found or not loaded in ModelStore.")
        self.upsampler = ctx.model

        # Handle input types like the image classifier 
        # if dir or file path is provided
        if isinstance(inputPath, str):
            # dir
            if os.path.isdir(inputPath):
                imagePaths = [
                    os.path.join(inputPath, f)
                    for f in os.listdir(inputPath)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
            #file    
            elif os.path.isfile(inputPath):
                if inputPath.lower().endswith(('.png', '.jpg', '.jpeg')):
                    imagePaths = [inputPath]
                else:
                    raise ValueError(f"Unsupported image file type: {inputPath}")
            else:
                raise FileNotFoundError(f"Input path does not exist: {inputPath}")
        # list 
        elif isinstance(inputPath, list):
            invalidPaths = [
                path for path in inputPath
                if not isinstance(path, str) or
                not os.path.isfile(path) or
                not path.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if invalidPaths:
                raise FileNotFoundError("Invalid or unsupported image files:\n" + "\n".join(invalidPaths))
            
            imagePaths = inputPath

        else:
            raise ValueError("Input must be a file path, directory path, or list of file paths")

        # Ensure output folder exists, creates the folder if does not exist
        os.makedirs(outputPath, exist_ok=True)

        enhancedOutputs = [] # store success only
        results = [] # store to see which image fails/sucess

        for path in imagePaths:
            try:
                img = Image.open(path).convert("RGB")
                img = np.array(img)

                with torch.no_grad():
                    output, _ = self.upsampler.enhance(img, outscale=1)

                base = os.path.splitext(os.path.basename(path))[0]
                ext = os.path.splitext(path)[1]
                outputPath = os.path.join(outputPath, f"{base}_upscaled{ext}")

                Image.fromarray(output).save(outputPath)

                tracer.addReport(ASReport(
                    source="REAL ESRGAN UPSCALE IMAGE",
                    message=f"Upscaled image saved to '{outputPath}'.",
                    extraData={"input": path, "output": outputPath}
                ))

                enhancedOutputs.append(outputPath) 

                results.append({"input": path, "output": outputPath, "status": "success"})

            except Exception as e:
                tracer.addReport(ASReport(
                    source="REAL ESRGAN UPSCALE IMAGE ERROR",
                    message=f"Failed to upscale '{path}': {e}",
                    extraData={"input": path, "exception": str(e)}
                ))

                results.append({"input": path, "error": str(e), "status": "error"})

        return {
            "successful": enhancedOutputs,
            "all_results": results
        } 

if __name__ == "__main__":
    ModelStore.setup(realESRGAN=RealESRGAN.loadModel)

    tracer = ArchSmith.newTracer("Test run of Real ESRGAN upscaling")

    RealESRGAN().upscaleImage(
        inputPath="Companydata/HF/02 Unidentified Photos (NYP)/1995 SCCCI 7251 receiving china vice premier_page-0001.jpg",  # folder or file or list
        outputPath="Companydata/Output",
        tracer=tracer
    )

    tracer.end()
    ArchSmith.persist()