project_name = 'UniAlignment'
run_name = 'pretrain'


json_lst = [
    ['ShareGPT-4o-Image_t2i.jsonl',
    'blip3o_t2i.jsonl'],
    ['ShareGPT-4o-Image_edit.jsonl',
     'uniworld_perception.jsonl'],
    ]

data_config = dict(resolution=512, max_length=256)

resume_from_legacy = "checkpoint/UniAlignment/transformer/diffusion_pytorch_model.safetensors"
pretrained_mask_emb = 'checkpoint/mask_token_emb.00-of-01.pth'
noise_scheduler_pretrained = 'stabilityai__stable-diffusion-3-medium-diffusers/scheduler'
sd3_pipeline_load_from = 'stabilityai__stable-diffusion-3-medium-diffusers'

training = dict(
    sampling_eps=1e-3, antithetic_sampling=True, importance_sampling=False, ignore_padding=False,
    caption_training_weight=0.2,
    contrastive_training_weight=0.05,
    alignment_training_weight=0.1,
)

# training setting
num_workers = 16
global_batch_size = 384
micro_batch_size = 4    

grad_clip = 2.0 

lr = 3.e-5
wd = 1.e-2
num_warmup_steps=2000
max_steps = 80000
ema_steps = 10000
log_every = 20
ckpt_every = 5000

train_real_cap_ratio = -1.0

ema_rate = 0.9995  
seed = 1234
disable_skip_data = True