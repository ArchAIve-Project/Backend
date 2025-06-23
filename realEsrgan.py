import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def upscale_image(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        model_path='models/RealESRGAN_x4plus.pth',  # Make sure this path exists!
        model=model,
        tile=128,           # Use tile > 0 if you run out of VRAM
        tile_pad=10,
        pre_pad=0,
        half= False,      # Change to True if using float16 with a modern GPU
        device=device
    )

    img = Image.open(input_path).convert('RGB')
    img = np.array(img)

    with torch.no_grad():
        output, _ = upsampler.enhance(img, outscale=1)  # outscale=1 means 4x internally

    Image.fromarray(output).save(output_path)
    print(f"Upscaled image saved to: {output_path}")

if __name__ == "__main__":
    input_img = 'Companydata/HF/02 Unidentified Photos (NYP)/1976 East Germany Delegation (4).jpg'
    output_img = 'Companydata/Output/1976 East Germany Delegation (4)_upscaled.png'
    upscale_image(input_img, output_img)
