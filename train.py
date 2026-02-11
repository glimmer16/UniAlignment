
"""
A minimal training script using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
import json
import logging
import os
import random
import socket
from time import time
import warnings
import gc
import sys
# byte-wandb huggingface
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
old_metadata = importlib_metadata.metadata

def new_metadata(name):
    if name == 'wandb':
        name =  'byted-wandb'
    return old_metadata(name)

importlib_metadata.metadata = new_metadata

from PIL import Image
from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from utils.grad_norm import calculate_l2_grad_norm, get_model_parallel_dim_dict, scale_grad
from utils.parallel import distributed_init, get_intra_node_process_group

import wandb
from mmcv import Config

from dataset.data_unify import FlexibleInternalData, FlexibleInternalData_edit, FlexibleInternalData_compose, FlexibleInternalData_url
from modules.loss_utils import ImageFlowMatchingLoss, TextMaskedDiffusionLoss, ImageFlowMatchingLoss_edit
from diffusers import StableDiffusion3Pipeline as SDPipe
from modules.unialignment_model import SD3JointModelFlexible

from safetensors.torch import save_file, load_file
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from torchvision.transforms import ToPILImage

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import cycle

import warnings
warnings.filterwarnings("ignore")  # ignore warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
MASK_TOKEN_IDS = 32099 # <'extra_id0'>, currently hard-coded

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999, copy=False):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())
    if not copy:
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    else:
        # for initialization

        for name, param in model_params.items():
            ema_params[name].data.copy_(param.data)

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def read_config(file):
    # solve config loading conflict when multi-processes
    import time
    while True:
        config = Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config

def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.encoder.block), # this assumes huggingface T5Encodermodel
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model

def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "h_sdp": ShardingStrategy._HYBRID_SHARD_ZERO2,
            "h_fsdp": ShardingStrategy.HYBRID_SHARD,
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model

def setup_mixed_precision(config):
    if config.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif config.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, hidden_states, encoder_hidden_states):
        img_feat = F.normalize(hidden_states.mean(dim=1), dim=-1)  # [B, 1536]
        txt_feat = F.normalize(encoder_hidden_states.mean(dim=1), dim=-1)  # [B, 1536]

        logits = img_feat @ txt_feat.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


def encode_img(processor, encoder_model, image):
    imgs = (image + 1) / 2.0      # image:  torch.Size([32, 3, 512, 512])
    imgs = imgs.clamp(0, 1)  
    image_inputs = []

    with torch.no_grad():
        for img in imgs:
            image_input = processor.image_processor.preprocess(images=img, return_tensors="pt", do_rescale=False).to(encoder_model.device)
            image_features = encoder_model.get_image_features(**image_input)
            image_inputs.append(image_features)

        image_inputs = torch.stack(image_inputs, dim=0)
        image_inputs = image_inputs.mean(dim=1)

    return image_inputs

def encode_texts(processor, encoder_model, texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(encoder_model.device)
    with torch.no_grad():
        text_embeds = encoder_model.get_input_embeddings()(inputs["input_ids"])
        text_features = text_embeds.mean(dim=1)
    
    return text_features

def cosine_similarity_loss(x, y):
    x_norm = F.normalize(x, p=2, dim=1) 
    y_norm = F.normalize(y, p=2, dim=1)
    cosine_sim = (x_norm * y_norm).sum(dim=1)
    loss = 1 - cosine_sim.mean()  
    return loss

class CombinedDataLoader:
    def __init__(self, dataset1, dataset2, local_batch_size, num_replicas, rank):
        self.batch_size = local_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        self.min_steps = min(len(dataset1) // (local_batch_size * num_replicas), len(dataset2) // (local_batch_size * num_replicas))
        
        self.sampler1 = DistributedSampler(dataset1, num_replicas=num_replicas, rank=rank, shuffle=True)
        self.sampler2 = DistributedSampler(dataset2, num_replicas=num_replicas, rank=rank, shuffle=True)
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
        self._init_loaders()
        
    def _init_loaders(self):
        self.sampler1.set_epoch(self.epoch)
        self.sampler2.set_epoch(self.epoch)
        self.epoch += 1
        
        self.loader1 = DataLoader(self.dataset1, batch_size=self.batch_size, sampler=self.sampler1, pin_memory=True, num_workers=min(16, os.cpu_count() // self.num_replicas), persistent_workers=True, drop_last=True)
        self.loader2 = DataLoader(self.dataset2, batch_size=self.batch_size, sampler=self.sampler2, pin_memory=True, num_workers=min(16, os.cpu_count() // self.num_replicas), persistent_workers=True, drop_last=True)
        
        self.iter1 = iter(self.loader1)
        self.iter2 = iter(self.loader2)
        self.flag_cycle = cycle([0, 1])
        self.steps = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.steps >= self.min_steps * 2: 
            self._init_loaders()
            self.steps = 0
            
        flag = next(self.flag_cycle)
        try:
            if flag == 0:
                batch = next(self.iter1)
            else:
                batch = next(self.iter2)
        except StopIteration as e:
            self._init_loaders()
            return self.__next__()
        
        self.steps += 1
        return {
            "type": "img_gen" if flag == 0 else "img_edit",
            "data": batch
        }

#############################################################################
#                                Training Loop                              #
#############################################################################

def main(args):
    """
    Trains a MMDiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    config = read_config(args.config)

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()
    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()

    assert config.global_batch_size % dp_world_size == 0, "Batch size must be divisible by data parrallel world size."
    local_batch_size = config.global_batch_size // dp_world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    if rank == 0:   
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        # initialize wandb
        wandb.init(project = config.project_name, name = config.run_name)
    else:
        logger = create_logger(None)

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))
    
     # load sd3 pipe
    sd_model_pipe = SDPipe.from_pretrained(config.sd3_pipeline_load_from, torch_dtype=torch.bfloat16).to(device) 
    sd_model_pipe.tokenizer_3.pad_token = sd_model_pipe.tokenizer_3.eos_token   # change padding token to eos token
    orig_sd_transformer = sd_model_pipe.transformer
    logger.info('sd pipeline loaded, text encoder was also prepared')
    
    model = SD3JointModelFlexible(len(sd_model_pipe.tokenizer_3), **orig_sd_transformer.config).train()
    logger.info(f"Model trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # remove unnecessary stuff from sd3 model pipe to save memory; delete all the encoder/weight not used in sd model pipe
    sd_model_pipe.transformer = None
    sd_model_pipe.text_encoder_1 = None
    sd_model_pipe.text_encoder_2 = None
    sd_model_pipe.text_encoder_3.requires_grad_(False) 
    sd_model_pipe.text_encoder_3.eval()
    
    def get_vae_feats(imgs):
        with torch.inference_mode():                            
            with torch.cuda.amp.autocast(enabled=True):  
                image_vae_feat = sd_model_pipe.vae.encode(imgs).latent_dist.sample()
                image_vae_feat = (image_vae_feat - sd_model_pipe.vae.config.shift_factor) * sd_model_pipe.vae.config.scaling_factor
        
            image_vae_feat = image_vae_feat.detach()
        return image_vae_feat

    # init multimodal encoder
    processor = AutoProcessor.from_pretrained(args.multimodal_encoder_path, trust_remote_code=True, use_fast=True)
    encoder_model = AutoModel.from_pretrained(args.multimodal_encoder_path, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().to(device)

    gc.collect()
    torch.cuda.empty_cache()

    if config.resume_from_legacy:
        resume_path = config.resume_from_legacy
        logger.info(f'Resuming legacy model from {resume_path}')
        
        state_dict = load_file(resume_path, device="cpu")
        missing, unexpect =  model.load_state_dict(state_dict, strict=False)

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpect}')
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f'Pretrained mdoel loaded from {resume_path}')
        
        if config.pretrained_mask_emb is not None:
            logger.info(f'Resuming t5 embedding from {config.pretrained_mask_emb}')
            mask_emb = torch.load(config.pretrained_mask_emb, map_location="cpu")
            sd_model_pipe.text_encoder_3.shared.weight.data[MASK_TOKEN_IDS] = mask_emb
            del mask_emb

    model_parallel_dim_dict = get_model_parallel_dim_dict(model)

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            logger.warning(f"Could not find existing checkpoints in {checkpoint_dir}")
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    if args.resume:
        logger.info(f"Resuming model weights from: {args.resume}")
        model.load_state_dict(load_file(os.path.join(args.resume, f"consolidated.{mp_rank:02d}-of-{mp_world_size:02d}.safetensors"), device="cpu"), strict=False)

    dist.barrier()

    model = setup_fsdp_sync(model, args)           

    # add t5 vocab embedding to optimizer
    opt = torch.optim.AdamW(list(model.parameters())+list(sd_model_pipe.text_encoder_3.get_input_embeddings().parameters()), lr=config.lr, weight_decay=config.wd)
    scheduler = get_constant_schedule_with_warmup(opt,
                                                num_warmup_steps=config.num_warmup_steps)
    if args.resume:
        opt_state_world_size = len([x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")])
        if opt_state_world_size != dist.get_world_size():
            logger.info(
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
            )
        else:
            logger.info(f"Resuming optimizer states from: {args.resume}")
            opt.load_state_dict(
                torch.load(os.path.join(args.resume, f"optimizer.{dist.get_rank():05d}-of-{dist.get_world_size():05d}.pth"), map_location="cpu")
            )
            for param_group in opt.param_groups:
                param_group["lr"] = config.lr
                param_group["weight_decay"] = config.wd

        # resume scheduler
        scheduler_state = torch.load(os.path.join(args.resume, f"scheduler.pth"), map_location="cpu")
        scheduler.load_state_dict(scheduler_state)

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:   
    logger.info("Creating dataset...")
    
    gen_dataset = FlexibleInternalData(json_lst=config.json_lst[0], tokenizer=sd_model_pipe.tokenizer_3, **config.data_config)
    edit_dataset = FlexibleInternalData_edit(json_lst=config.json_lst[1], tokenizer=sd_model_pipe.tokenizer_3, **config.data_config)
    #gen_url_dataset = FlexibleInternalData_url(json_lst=config.json_lst[0], tokenizer=sd_model_pipe.tokenizer_3, **config.data_config)
    #compose_dataset = FlexibleInternalData_compose(json_lst=config.json_lst[1], tokenizer=sd_model_pipe.tokenizer_3, **config.data_config)

    gen_loss_module = ImageFlowMatchingLoss(model_pipe = sd_model_pipe, text_max_length=256, )
    edit_loss_module = ImageFlowMatchingLoss_edit(model_pipe = sd_model_pipe, text_max_length=256,)

    #t2i_data_iter = CombinedDataLoader(gen_url_dataset, edit_dataset, local_batch_size=int(local_batch_size), num_workers=config.num_workers)
    t2i_data_iter = CombinedDataLoader(gen_dataset, edit_dataset, local_batch_size=int(local_batch_size), num_replicas=dist.get_world_size(), rank=rank)

    logger.info('Dataset created')
    logger.info(f'{config.data_config}')

    text_diffusion_loss_module = TextMaskedDiffusionLoss(config, model_pipe = sd_model_pipe, )
    contrastive_loss_module = ContrastiveLoss()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_caption_loss = running_image_loss = running_contrastive_loss = running_text_align_loss = running_img_align_loss = 0.0
    start_time = time()
    step = 0
    if not config.disable_skip_data:
        logger.info(f'Skipping {resume_step} steps in dataloader')
        #if rank == 0:   
        #    print(f'Skipping {resume_step} steps in dataloader')
        while step < resume_step:
            if step % 100 ==0:
                logger.info(f'Skipped step {step} in dataloader')
                gc.collect()
                torch.cuda.empty_cache()
            try:
                _ = next(t2i_data_iter)
            except StopIteration:
                logger.info('DataLoader exhausted while skipping steps, reinitializing iterator')
            step += 1
    else:
        step = resume_step


    logger.info(f"Training for {config.max_steps:,} steps...")
    
    while step < config.max_steps:
        try:
            batch = next(t2i_data_iter)
            
        except RuntimeError as e:
            if "pin memory" in str(e):
                print(f"Rank {rank}: Pin memory error, skipping batch")
                continue
            else:
                raise e
            
        data_type = batch["type"]
        t2i_batch = batch["data"]

        if data_type == "img_gen" or data_type == "img_gen_url":
            t2i_imgs = t2i_batch[0].to(device, non_blocking=True)       # gen
            i2t_imgs = t2i_batch[0].to(device, non_blocking=True)       # und
            i2t_caption = t2i_batch[1]                                  # caption
            instruction_ids = t2i_batch[2]['input_ids'].to(device).squeeze()    # caption_id_gen
            caption_ids = t2i_batch[2]['input_ids'].to(device).squeeze()    # caption_id_und
            
            t2i_vae_feats, i2t_vae_feats = get_vae_feats(t2i_imgs), get_vae_feats(i2t_imgs)     # vae_gen; vae_und

        elif data_type == "img_edit":
            t2i_imgs = t2i_batch[0].to(device, non_blocking=True)       # source 
            i2t_imgs = t2i_batch[1].to(device, non_blocking=True)       # edited
            i2t_caption = t2i_batch[2]                                  # caption
            caption_ids = t2i_batch[3]['input_ids'].to(device).squeeze()    # caption_id
            instruction_ids = t2i_batch[4]['input_ids'].to(device).squeeze()    # instruction_id
            
            t2i_vae_feats, i2t_vae_feats = get_vae_feats(t2i_imgs), get_vae_feats(i2t_imgs)

        elif data_type == "img_compose":
            t2i_imgs_1 = t2i_batch[0].to(device, non_blocking=True)         # cond_1_2_3
            t2i_imgs_2 = t2i_batch[1].to(device, non_blocking=True)
            t2i_imgs_3 = t2i_batch[2].to(device, non_blocking=True)
            i2t_imgs = t2i_batch[3].to(device, non_blocking=True)           # edited
            i2t_caption = t2i_batch[4]                                      # caption
            caption_ids = t2i_batch[5]['input_ids'].to(device).squeeze() # caption_id
            instruction_ids = t2i_batch[6]['input_ids'].to(device).squeeze() # instruction_id

            t2i_vae_feats_1, t2i_vae_feats_2, t2i_vae_feats_3, i2t_vae_feats = get_vae_feats(t2i_imgs_1), get_vae_feats(t2i_imgs_2), get_vae_feats(t2i_imgs_3), get_vae_feats(i2t_imgs)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                img_siglip_feats = encode_img(processor, encoder_model, i2t_imgs).detach()          # torch.Size([32, 3584])
                text_siglip_feats = encode_texts(processor, encoder_model, i2t_caption).detach() 
                
        loss_item = 0.0
        opt.zero_grad()
        
        # the micro batch size controls gradient accumulation
        for mb_idx in range((local_batch_size - 1) // config.micro_batch_size + 1):
            
            mb_st = mb_idx * config.micro_batch_size
            mb_ed = min((mb_idx + 1) * config.micro_batch_size, local_batch_size)
            last_mb = mb_ed == local_batch_size

            if data_type == "img_gen" or data_type == "img_gen_url" or data_type == "img_edit":
                t2i_imgs_mb =  t2i_vae_feats[mb_st:mb_ed]
            elif data_type == "img_compose":
                t2i_imgs_mb_1 =  t2i_vae_feats_1[mb_st:mb_ed]
                t2i_imgs_mb_2 =  t2i_vae_feats_2[mb_st:mb_ed]
                t2i_imgs_mb_3 =  t2i_vae_feats_3[mb_st:mb_ed]
                t2i_imgs_mb = [t2i_imgs_mb_1, t2i_imgs_mb_2, t2i_imgs_mb_3]

            i2t_imgs_mb =   i2t_vae_feats[mb_st:mb_ed]
            i2t_ids_mb = caption_ids[mb_st:mb_ed]
            t2i_ids_mb = instruction_ids[mb_st:mb_ed]
            text_siglip_mb = text_siglip_feats[mb_st:mb_ed]
            img_siglip_mb = img_siglip_feats[mb_st:mb_ed]

            with {"bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16), "fp16": torch.cuda.amp.autocast(dtype=torch.float16), "fp32": contextlib.nullcontext(), "tf32": contextlib.nullcontext()}[args.precision]:
                with torch.inference_mode():
                    t2i_emb_mb = sd_model_pipe.text_encoder_3(t2i_ids_mb)[0].detach()

                # image loss
                if data_type == "img_gen" or data_type == "img_gen_url":
                    image_diffusion_loss, hidden_states, mid_hidden_states = gen_loss_module.compute_loss(
                                                        model, t2i_emb_mb, t2i_imgs_mb, block_depth=args.block_depth,
                                                    )
                elif data_type == "img_edit" or data_type == "img_compose":
                    image_diffusion_loss, hidden_states, mid_hidden_states = edit_loss_module.compute_loss(
                                                        model, t2i_emb_mb, t2i_imgs_mb, edit_imgs=i2t_imgs_mb, data_type=data_type, block_depth=args.block_depth,
                                                    )

                # caption loss
                caption_diffusion_loss, encoder_hidden_states, mid_encoder_hidden_states = text_diffusion_loss_module.compute_loss(
                                                    model, i2t_ids_mb, i2t_imgs_mb, None,
                                                    use_dummy_loss=False, disable_t5_grad=True, block_depth=args.block_depth,
                                                )
                contrastive_loss = contrastive_loss_module(hidden_states, encoder_hidden_states)

                loss_text_align = cosine_similarity_loss(mid_encoder_hidden_states, text_siglip_mb)
                loss_img_align = cosine_similarity_loss(mid_hidden_states, img_siglip_mb)

                loss = image_diffusion_loss + caption_diffusion_loss * config.training.caption_training_weight + \
                    config.training.contrastive_training_weight * contrastive_loss + \
                    config.training.alignment_training_weight * (loss_text_align + loss_img_align)

            with model.no_sync() if args.data_parallel in ['h_sdp', 'h_fsdp', "sdp", "fsdp"] and not last_mb else contextlib.nullcontext():
                loss.backward()

            running_caption_loss += caption_diffusion_loss.item()
            running_image_loss += image_diffusion_loss.item()
            running_contrastive_loss += contrastive_loss.item()
            running_text_align_loss += loss_text_align.item()
            running_img_align_loss += loss_img_align.item()
                
            loss_item += loss.item()

        grad_norm = calculate_l2_grad_norm(model, model_parallel_dim_dict)
        if grad_norm > config.grad_clip:
            scale_grad(model, config.grad_clip / grad_norm)        

        opt.step()
        scheduler.step()

        step += 1

        # Log loss values:
        running_loss += loss_item
        log_steps += 1
        if step % config.log_every == 0:
            gradient_accumulation_steps = (local_batch_size - 1) // config.micro_batch_size + 1
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = config.global_batch_size * log_steps / (end_time - start_time)

            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_caption_loss = torch.tensor(running_caption_loss / log_steps, device=device)
            dist.all_reduce(avg_caption_loss, op=dist.ReduceOp.SUM)
            avg_caption_loss = avg_caption_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_image_loss = torch.tensor(running_image_loss / log_steps, device=device)
            dist.all_reduce(avg_image_loss, op=dist.ReduceOp.SUM)
            avg_image_loss = avg_image_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_contrastive_loss = torch.tensor(running_contrastive_loss / log_steps, device=device)
            dist.all_reduce(avg_contrastive_loss, op=dist.ReduceOp.SUM)
            avg_contrastive_loss = avg_contrastive_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_text_align_loss = torch.tensor(running_text_align_loss / log_steps, device=device)
            dist.all_reduce(avg_text_align_loss, op=dist.ReduceOp.SUM)
            avg_text_align_loss = avg_text_align_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            avg_img_align_loss = torch.tensor(running_img_align_loss / log_steps, device=device)
            dist.all_reduce(avg_img_align_loss, op=dist.ReduceOp.SUM)
            avg_img_align_loss = avg_img_align_loss.item() / dist.get_world_size() / gradient_accumulation_steps

            logger.info(
                f"(Step={step + 1:07d}) "
                f"Image Loss: {avg_image_loss:.4f}, "
                f"Caption Loss: {avg_caption_loss:.4f}, "
                f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
                f"Img Align Loss: {avg_img_align_loss:.4f}, "
                f"Text Align Loss: {avg_text_align_loss:.4f}, "
                f"Train Secs/Step: {secs_per_step:.2f}, "
                f"Train Imgs/Sec: {imgs_per_sec:.2f}, "
                f"Train grad norm: {grad_norm:.2f},"
            )

            # Reset monitoring variables:
            running_loss = running_image_loss = running_caption_loss = running_contrastive_loss = running_text_align_loss = running_img_align_loss = 0

            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if step % config.ckpt_every == 0 or step == config.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True, offload_to_cpu=True)):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (f"consolidated.{fs_init.get_model_parallel_rank():02d}-of-{fs_init.get_model_parallel_world_size():02d}.safetensors")
                    save_file(consolidated_model_state_dict, os.path.join(checkpoint_path, consolidated_fn), metadata={"step": str(step+1), "format": "FSDP-sharded"})
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT,):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            # just save scheduler on the main rank
            if rank == 0:
                scheduler_state_fn = f"scheduler.pth"
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, scheduler_state_fn))
            dist.barrier()
            logger.info(f"Saved scheduler to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--multimodal_encoder_path", type=str, default="Qwen__Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--block_depth", type=int, default=8)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume", help="Do NOT auto resume from the last checkpoint in --results_dir.")
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["h_sdp", "h_fsdp", "sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"], default='fp32')
    
    parser.add_argument("--global_seed", type=int, default=971011)
    args = parser.parse_args()

    main(args)