import os
import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from addons import ASTracer, ASReport, ArchSmith

class RealESRGAN:
    def __init__(self):
        DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        self.upsampler = RealESRGANer(
            scale=4,
            model_path='models/RealESRGAN_x4plus.pth',
            model=model,
            tile=128,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=DEVICE
        )

    def upscale_image(self, input_path, output_dir: str, tracer: ASTracer):
        # Handle input types like the image classifier 
        # if dir or file path is provided
        if isinstance(input_path, str):
            # dir
            if os.path.isdir(input_path):
                image_paths = [
                    os.path.join(input_path, f)
                    for f in os.listdir(input_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
            #file    
            elif os.path.isfile(input_path):
                if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths = [input_path]
                else:
                    raise ValueError(f"Unsupported image file type: {input_path}")
            else:
                raise FileNotFoundError(f"Input path does not exist: {input_path}")
        # list 
        elif isinstance(input_path, list):
            invalid_paths = [
                path for path in input_path
                if not isinstance(path, str) or
                not os.path.isfile(path) or
                not path.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if invalid_paths:
                raise FileNotFoundError("Invalid or unsupported image files:\n" + "\n".join(invalid_paths))
            
            image_paths = input_path

        else:
            raise ValueError("Input must be a file path, directory path, or list of file paths")

        # Ensure output folder exists
        os.makedirs(output_dir, exist_ok=True)

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = np.array(img)

                with torch.no_grad():
                    output, _ = self.upsampler.enhance(img, outscale=1) # outscale=1 means 4x internally

                base = os.path.splitext(os.path.basename(path))[0]
                ext = os.path.splitext(path)[1]
                output_path = os.path.join(output_dir, f"{base}_upscaled{ext}") # outpath = same name + _upscaled

                Image.fromarray(output).save(output_path)

                tracer.addReport(ASReport(
                    source="REALESRGAN UPSCALE_IMAGE",
                    message=f"Upscaled image saved to '{output_path}'.",
                    extraData={"input": path, "output": output_path}
                ))

            except Exception as e:
                tracer.addReport(ASReport(
                    source="REALESRGAN UPSCALE_IMAGE ERROR",
                    message=f"Failed to upscale '{path}': {e}",
                    extraData={"input": path, "exception": str(e)}
                ))

if __name__ == "__main__":
    tracer = ArchSmith.newTracer("Test run of RealESRGAN upscaling")

    real_esrgan = RealESRGAN()
    real_esrgan.upscale_image(
        input_path="Companydata/HF/Tan Keong Choon/27.8.95国庆日国家古迹之行.jpg",  # folder or file or list
        output_dir="Companydata/Output",
        tracer=tracer
    )

    tracer.end()
    ArchSmith.persist()