import os
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from modules.pipeline import UniAlignmentPipeline
import numpy as np
import torch
from PIL import Image

device = 'cuda'

def img_gen(pipe, task):
    
    prompt = "A detailed photograph captures the intricate features of a pharaoh statue adorned with unconventional accessories. The statue is wearing steampunk glasses that have intricate bronze gears and round, reflective lenses. It is also dressed in a stark white t-shirt that contrasts with a dark, textured leather jacket draped over its shoulders. The image is taken with a high-quality DSLR camera, ensuring that the textures and colors of the statue and its attire are vivid and sharp. The background is a simple, unobtrusive blur, drawing all attention to the anachronistic ensemble of the pharaoh."
    save_path = './outputs/t2i_result.png'
    
    pipe.set_sampling_mode(task)
    imgs = pipe(
            prompt=prompt,
            negative_prompt='low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame',
            height=512,
            width=512,
            num_inference_steps=50,
            num_images_per_prompt=1,
            data_type=task)

    sample = torch.clamp(imgs[0].float()*0.5+0.5, 0, 1).cpu()
    sample_np = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(sample_np)
    image.save(save_path)

    return

def img_edit(pipe, task):

    prompt = "Transform the illustration into an 8-bit pixel-art style suitable for a video game."
    img_path = "./assests/images/1.jpg" 
    img = Image.open(img_path)
    save_path = './outputs/edit_result.png'

    pipe.set_sampling_mode(task)
    imgs = pipe(
            prompt=prompt,
            negative_prompt='low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame',
            img=img,
            height=512,
            width=512,
            num_images_per_prompt=1,
            data_type=task)
    
    sample = torch.clamp(imgs[0].float()*0.5+0.5, 0, 1).cpu()
    sample_np = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(sample_np)

    image.save(save_path)

    return

def img_perception(pipe, task):

    prompt = "Provide a detailed depth map of the entire scene."
    img_path = "./assests/images/2.jpg" 
    img = Image.open(img_path)
    save_path = './outputs/perception_result.png'

    pipe.set_sampling_mode(task)
    imgs = pipe(
            prompt=prompt,
            negative_prompt='low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame',
            img=img,
            height=512,
            width=512,
            num_images_per_prompt=1,
            data_type=task)
    
    sample = torch.clamp(imgs[0].float()*0.5+0.5, 0, 1).cpu()
    sample_np = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    image = Image.fromarray(sample_np)

    image.save(save_path)

    return

def img_caption(pipe, task):
   
    img_path = "./assests/images/3.jpg"
    img = Image.open(img_path)
    
    pipe.set_sampling_mode(task)
    caption = pipe(image=img, prompt=None, sequence_length=256, num_inference_steps=32, resolution=512)
    print(caption)

    return


if __name__ == "__main__":

    pipe = UniAlignmentPipeline.from_pretrained("./checkpoint/UniAlignment", torch_dtype=torch.bfloat16).to(device)
    
    os.makedirs('./outputs', exist_ok=True)

    img_gen(pipe, "img_gen")
    print("Image generation completed. Check the generated image at '.outputs/t2i_result.png'.")

    img_edit(pipe, "img_edit")
    print("Image editing completed. Check the generated image at '.outputs/edit_result.png'.")

    img_perception(pipe, "img_edit")
    print("Image editing completed. Check the generated image at '.outputs/perception_result.png'.")

    img_caption(pipe, "i2t")
    print("Image captioning completed. Check the console for the generated caption.")
