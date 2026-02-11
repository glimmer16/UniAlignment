import os
import random
from PIL import Image
import json
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
import requests
from io import BytesIO


def load_json(file_path):
    with open(file_path, 'r') as f:
        meta_data = json.load(f)

    return meta_data


class FlexibleInternalData_Llava_pretrained(Dataset):
    def __init__(self,
                 roots,       # a list of root that has the same length as image_list_json_lst
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=512,
                 org_caption_key=None,
                 re_caption_key=None,
                 tokenizer=None,
                 max_length=None,
                 **kwargs):

        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.org_captions = []

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        if not isinstance(org_caption_key, list):
            org_caption_key = [org_caption_key] * len(json_lst)
        
        for root, json_file in zip(roots, json_lst):
            meta_data = load_json(os.path.join(root, json_file))
            meta_data_clean = [item for item in meta_data]
            self.img_samples.extend([os.path.join(root, item['image']) for item in meta_data_clean])
            self.org_captions.extend([item['conversations'][1]['value'] for item in meta_data_clean])

        self.loader = default_loader

    def getdata(self, index):
        img_path = self.img_samples[index]
        origin_caption = self.org_captions[index]

        data_info = {}
        raw_img = self.loader(img_path)
        h, w = (raw_img.size[1], raw_img.size[0])
        data_info['img_hw'] = torch.tensor([h, w], dtype=torch.float32) 
        img_tsr = self.transform(raw_img)

        if self.tokenizer is not None:         
            org_caption_token_info = self.tokenizer(origin_caption, padding="max_length", truncation=True, max_length=self.max_length, eturn_tensors="pt", )

            return img_tsr, origin_caption, org_caption_token_info
        else:
            return img_tsr, origin_caption, origin_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')



class FlexibleInternalData(Dataset):
    def __init__(self,
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=512,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):

        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.org_captions = []

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        for json_file in json_lst:
            with open(json_file, 'r', encoding='utf-8') as file:
                for meta_data in file:
                    item = json.loads(meta_data.strip())
                    self.img_samples.extend([item['image'][0]])
                    self.org_captions.extend([item['conversations'][0]['value']])

        self.loader = default_loader

    def getdata(self, index):
        img_path = self.img_samples[index]
        origin_caption = self.org_captions[index]   # caption

        raw_img = self.loader(img_path)
        img_tsr = self.transform(raw_img)
        if self.tokenizer is not None:         
            org_caption_token_info = self.tokenizer(origin_caption,padding="max_length",truncation=True,max_length=self.max_length,return_tensors="pt",)
            return img_tsr, origin_caption, org_caption_token_info        
        else:
            return img_tsr, origin_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')


class FlexibleInternalData_url(Dataset):
    def __init__(self,
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=512,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):

        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.org_captions = []

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        for json_file in json_lst:
            with open(json_file, 'r', encoding='utf-8') as file:
                for meta_data in file:
                    item = json.loads(meta_data.strip())
                    self.img_samples.extend([item['image'][0]])
                    self.org_captions.extend([item['conversations'][0]['value']])

        self.loader = default_loader

    def getdata(self, index):
        img_path = self.img_samples[index]

        if img_path.startswith("http"):     
            try:               
                response = requests.get(img_path)
                response.raise_for_status()
                
                image_data = BytesIO(response.content)
                raw_img = Image.open(image_data).convert("RGB")
            except Exception as e1:
                raise Exception(f"wrong image http path: {img_path}, error info: {e1}")

        origin_caption = self.org_captions[index]   # caption
        img_tsr = self.transform(raw_img)

        if self.tokenizer is not None:       
            org_caption_token_info = self.tokenizer(origin_caption,padding="max_length",truncation=True,max_length=self.max_length,return_tensors="pt",)
            return img_tsr, origin_caption, org_caption_token_info
        else:
            return img_tsr, origin_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')


class FlexibleInternalData_edit(Dataset):
    def __init__(self,
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=512,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):

        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.edit_samples = []
        self.org_captions = []
        self.edit_captions = []
        self.loader = default_loader

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        for json_file in json_lst:
            with open(json_file, 'r', encoding='utf-8') as file:
                for meta_data in file:
                    item = json.loads(meta_data.strip())
                    self.img_samples.extend([item['image'][0]])
                    self.edit_samples.extend([item['image'][1]])
                    self.org_captions.extend([item['conversations'][0]['value']])
                    self.edit_captions.extend([item['instruction']])

    def getdata(self, index):
        img_path = self.img_samples[index]
        edit_path = self.edit_samples[index]
        origin_caption = self.org_captions[index]   # caption
        edit_caption = self.edit_captions[index]    # instruction

        raw_img = self.loader(img_path)
        img_tsr = self.transform(raw_img)
        raw_edit = self.loader(edit_path)
        edit_tsr = self.transform(raw_edit)

        if self.tokenizer is not None:        
            # tokenize
            org_caption_token_info = self.tokenizer(origin_caption,padding="max_length",truncation=True,max_length=self.max_length,return_tensors="pt",)
            edit_caption_token_info = self.tokenizer(edit_caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt", )
            return img_tsr, edit_tsr, origin_caption, org_caption_token_info, edit_caption_token_info
        
        else:
            return img_tsr, edit_tsr, origin_caption, edit_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')


class FlexibleInternalData_compose(Dataset):
    def __init__(self,
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=256,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):

        self.resolution = resolution

        self.meta_data_clean = []
        self.img_samples_1 = []
        self.img_samples_2 = []
        self.img_samples_3 = []
        self.edit_samples = []
        self.org_captions = []
        self.re_captions = []
        self.edit_captions = []

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        for json_file in json_lst:
            with open(json_file, 'r', encoding='utf-8') as file:
                for meta_data in file:
                    item = json.loads(meta_data.strip())
                    self.img_samples_1.extend([item['original_images'][0]])
                    self.img_samples_2.extend([item['original_images'][1]])
                    self.img_samples_3.extend([item['original_images'][2]])
                    self.edit_samples.extend([item['generated_images'][0]])
                    self.edit_captions.extend([item['edit_prompt'].strip(' \\"')])
                    self.org_captions.extend([item['caption']])

        self.loader = default_loader

    def getdata(self, index):
        img_path_1 = self.img_samples_1[index]
        img_path_2 = self.img_samples_2[index]
        img_path_3 = self.img_samples_3[index]
        edit_path = self.edit_samples[index]
        origin_caption = self.org_captions[index]
        
        edit_caption = self.edit_captions[index]
        
        raw_img_1 = self.loader(img_path_1)
        img_tsr_1 = self.transform(raw_img_1)
        raw_img_2 = self.loader(img_path_2)
        img_tsr_2 = self.transform(raw_img_2)
        raw_img_3 = self.loader(img_path_3)
        img_tsr_3 = self.transform(raw_img_3)

        raw_edit = self.loader(edit_path)
        edit_tsr = self.transform(raw_edit)

        if self.tokenizer is not None: 
            # tokenize
            org_caption_token_info = self.tokenizer(origin_caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            edit_caption_token_info = self.tokenizer(edit_caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

            return img_tsr_1, img_tsr_2, img_tsr_3, edit_tsr, origin_caption, org_caption_token_info, edit_caption_token_info
        
        else:
            return edit_tsr, edit_tsr, origin_caption, edit_caption
        
    def __len__(self):
        return len(self.edit_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')





class FlexibleInternalData_mixed(Dataset):
    def __init__(self,
                 type,       # a list of root that has the same length as image_list_json_lst
                 json_lst=None,   # a list of json file, each json file contains a list of dict, each dict contains the info of an image and its caption
                 resolution=512,
                 tokenizer=None, # do tokenizing on the fly
                 max_length=None,
                 **kwargs):

        self.resolution = resolution
        self.meta_data_clean = []
        self.img_samples = []
        self.edit_samples = []
        self.org_captions = []
        self.edit_captions = []
        self.loader = default_loader

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            assert max_length is not None, "max_length must be provided when tokenizer is not None"
            self.max_length = max_length

        self.interpolate_model = InterpolationMode.BICUBIC
        self.transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB')),
                T.Resize(self.resolution, interpolation=self.interpolate_model),  # Image.BICUBIC
                T.CenterCrop(self.resolution),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])

        for data_type, json_file in zip(type, json_lst):
            if data_type == 'img_edit':
                with open(json_file, 'r', encoding='utf-8') as file:
                    for meta_data in file:
                        item = json.loads(meta_data.strip())
                        self.img_samples.extend([item['image'][0]])
                        self.edit_samples.extend([item['image'][1]])
                        self.org_captions.extend([item['conversations'][0]['value']])
                        self.edit_captions.extend([item['instruction']])
            elif data_type == 'img_gen':
                with open(json_file, 'r', encoding='utf-8') as file:
                    for meta_data in file:
                        item = json.loads(meta_data.strip())
                        self.img_samples.extend([item['image'][0]])
                        self.org_captions.extend([item['conversations'][0]['value']])


    def getdata(self, index):
        img_path = self.img_samples[index]
        edit_path = self.edit_samples[index]
        origin_caption = self.org_captions[index]   # caption
        edit_caption = self.edit_captions[index]    # instruction

        raw_img = self.loader(img_path)
        img_tsr = self.transform(raw_img)
        raw_edit = self.loader(edit_path)
        edit_tsr = self.transform(raw_edit)

        if self.tokenizer is not None:         
            # tokenize
            org_caption_token_info = self.tokenizer(origin_caption,padding="max_length",truncation=True,max_length=self.max_length,return_tensors="pt",)
            edit_caption_token_info = self.tokenizer(edit_caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt", )
            return img_tsr, edit_tsr, origin_caption, org_caption_token_info, edit_caption_token_info
        
        else:
            return img_tsr, edit_tsr, origin_caption, edit_caption
        
    def __len__(self):
        return len(self.img_samples)

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.__len__) # get a closest
        raise RuntimeError('Too many bad data.')