import math, os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import datetime
import sys
import time
import types
import warnings
import sys
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from modules.pipeline import UniAlignmentPipeline

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
from modules.unialignment_model import SD3JointModelFlexible
from diffusers import StableDiffusion3Pipeline as SDPipe
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import json

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    new_area = width * height
    if new_area < target_area:
        width += 32
        new_area = width * height
    elif new_area > target_area:
        width -= 32
        new_area = width * height
    
    return width, height, new_area

device = 'cuda'
save_path = "benchmark_results"

transformer = SD3JointModelFlexible.from_pretrained("checkpoint/UniAlignment/transformer")
pipe = UniAlignmentPipeline.from_pretrained("checkpoint/UniAlignment", transformer=transformer, torch_dtype=torch.bfloat16).to(device)


json_file = "benchmark/semgen_bench.jsonl"
with open(json_file, 'r', encoding='utf-8') as file:
    for key, meta_data in enumerate(file):
        item = json.loads(meta_data.strip())
        img_path = item['image_path']

        input_img_raw = Image.open(img_path)
        print(key)
        resize_input_image = input_img_raw.resize((512, 512))
    
        for task_type in ["t2i_long", "t2i_complex"]: 
            save_path_fullset = f"{save_path}/fullset/{task_type}/{key}.png"
            os.makedirs(os.path.dirname(save_path_fullset), exist_ok=True)

            pipe.set_sampling_mode('img_gen')
            imgs_long = pipe(
                prompt=item[task_type],
                negative_prompt='low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame',
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=7.0,
                num_images_per_prompt=1,
                data_type='img_gen')
            sample = torch.clamp(imgs_long[0].float()*0.5+0.5, 0, 1).cpu()
            sample_np = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(sample_np)
            image.save(save_path_fullset)
    
        for task_type in ["edit_multi", "edit_complex"]:
            save_path_fullset_source_image = f"{save_path}/fullset/{task_type}/{key}_SRCIMG.png"
            save_path_fullset = f"{save_path}/fullset/{task_type}/{key}.png"
            os.makedirs(os.path.dirname(save_path_fullset_source_image), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_fullset), exist_ok=True)
            
            pipe.set_sampling_mode('img_edit')
            imgs = pipe(
                prompt=item[task_type],
                negative_prompt='low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame',
                img=resize_input_image,
                height=512,
                width=512,
                num_inference_steps=50,
                guidance_scale=9.0,
                num_images_per_prompt=1,
                data_type='img_edit')
            sample = torch.clamp(imgs[0].float()*0.5+0.5, 0, 1).cpu()
            sample_np = (sample.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            image = Image.fromarray(sample_np)
            image.save(save_path_fullset)
            resize_input_image.save(save_path_fullset_source_image)

print("SemGen-Bench test done.")
