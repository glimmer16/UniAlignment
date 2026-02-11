# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from webdatasets."""

import os
import numpy as np
import logging
import PIL.Image
from functools import partial
from pathlib import Path
from glob import glob

import torch
import webdataset as wds
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def log_and_continue(exn) -> bool:
    logging.warning(f'Webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(
    data,
    keys=base_plus_ext,
    lcase=True,
    suffixes=None,
    handler=None,
):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # Re-implementation of the wds with group_by_keys that doesn't throw.
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler, eof_value=None)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def preprocess_img(img: PIL.Image, resolution: int = 256) -> np.ndarray:
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis] # HW => HWC
    img = center_crop(resolution, resolution,  img)
    img = img.transpose(2, 0, 1) # HWC => CHW
    img = img.astype(np.float32) / 255.0
    img = 2.0 * img - 1.0
    img = torch.from_numpy(img)
    return img


def preprocess_txt(text: str) -> str:
    return text

def preprocess_extract_txt(info: dict, text_tokenizer, max_length) -> str:
    re_cap, org_cap = info['caption'], info['org_caption']
    recap_token_info = text_tokenizer(re_cap, truncation=True, max_length=max_length,
                                            padding="max_length",  return_tensors="pt")
    orgcap_token_info = text_tokenizer(org_cap, truncation=True, max_length=max_length,
                                            padding="max_length",  return_tensors="pt")
    return recap_token_info, orgcap_token_info

def filter_no_caption(sample: dict) -> bool:
    return 'txt' in sample

def filter_image_shape_and_ar(sample: dict) -> bool:
    img = sample['image']
    img_w, img_h = img.size
    img_ar = img_w / img_h
    return min(img_h, img_w) >= 100 and img_ar <= 4.0 and img_ar >= 1.0 / 4.0



def center_crop(width: int, height: int, img: np.ndarray) -> np.ndarray:
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
    img = PIL.Image.fromarray(img, 'RGB')
    img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
    
    return np.array(img)


def custom_collate_fn(batch):
    # Handle variable-length sequences here, e.g., by padding
    # Return the batch in the desired format
    # stack images, concat keys in info
    images = torch.stack([item[0] for item in batch])
    # stack batch encodings
    re_captions = {k: torch.cat([item[1][0][k] for item in batch]) for k in ['input_ids', 'attention_mask']}
    org_captions = {k: torch.cat([item[1][1][k] for item in batch]) for k in ['input_ids', 'attention_mask']}
    return images, re_captions, org_captions


def get_urls(path: str) -> list[str]:
    '''
    Expected file structure:
    path/
    path/part1/0000.tar
    path/part1/0001.tar
    ...
    path/part2/0000.tar
    path/part2/0001.tar
    ...

    Dataloader can be used while the dataset is still downloading,
    only fully downloaded tars will be used.
    '''
    fpaths = glob(f'{path}/*.json') # in this way, we only use fullydownloaded datasets
    fpaths.sort()
    urls = [p.replace('_stats.json', '.tar') for p in fpaths]
    return urls


def wds_dataloader(
    train_data: list[str],
    text_tokenizer,
    *,
    batch_size: int, # this will be per device
    resolution: int,
    max_length: int,
    workers: int = 3,
    shard_shuffle_size: int = 1000,
    sample_shuffle_size: int = 10000,
) -> wds.WebLoader:
    input_shards = train_data
    assert input_shards is not None

    dataset = wds.DataPipeline([
        wds.ResampledShards(input_shards, deterministic=True),
        tarfile_to_samples_nothrow,
        wds.shuffle(shard_shuffle_size, seed=6666),
        # wds.select(filter_no_caption),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.shuffle(sample_shuffle_size, seed=6666),
        wds.rename(image="jpg;png", info="json"),
        wds.select(filter_image_shape_and_ar),
        wds.map_dict(image=partial(preprocess_img, resolution=resolution), info=partial(preprocess_extract_txt, text_tokenizer=text_tokenizer, max_length=max_length)),
        wds.to_tuple("image", "info"),
        wds.batched(batch_size, custom_collate_fn),
    ])

    # build dataloader
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,

    )

    return dataloader


class WdsWrapper():
    def __init__(
        self,
        path: str,
        resolution: int,
        **kwargs,
    ):
        self._path = Path(path)
        self._name = os.path.splitext(os.path.basename(self._path))[0]
        self.resolution = resolution
        self.urls = self.get_urls(path)

        # Only using WDS for text datasets.
        self.labels_are_text = True
        self.has_labels = True

        # Load a few images for visualization.
        self.first_images = []
        self.first_labels = []
        self.init_viz()

    def get_urls(self, path: str) -> list[str]:
        '''
        Expected file structure:
        path/
        path/part1/0000.tar
        path/part1/0001.tar
        ...
        path/part2/0000.tar
        path/part2/0001.tar
        ...

        Dataloader can be used while the dataset is still downloading,
        only fully downloaded tars will be used.
        '''
        fpaths = glob(f'{path}/*.json')
        urls = [p.replace('_stats.json', '.tar') for p in fpaths]
        return urls

    def init_viz(self) -> None:
        gw = np.clip(7680 // self.resolution, 7, 32)
        gh = np.clip(4320 // self.resolution, 4, 32)
        dl = iter(wds_dataloader(self.urls, batch_size=gw*gh, workers=1, resolution=self.resolution))
        self.first_images, self.first_labels = next(dl)

    def __len__(self) -> int:
        return len(self.urls) * 10000

    @property
    def image_shape(self) -> list[int]:
        return [3, self.resolution, self.resolution]

    @property
    def label_shape(self) -> list[int]:
        return [1]

    @property
    def label_dim(self) -> int:
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str] :
        return self.first_images[idx], self.first_labels[idx]