from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from bra_gen import generate_dataset
from bra_ocr import *
import torch

#generate_dataset('bra16', 16, 8)

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 100,           # number of steps
    sampling_timesteps = 100    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    '/cluster/home1/lurui/ddpm_bra/bra16',
    train_batch_size = 16,
    results_folder = 'exp_bra16',
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False             # whether to calculate fid during training
)

trainer.load(80)

trainer.ema.ema_model.eval()
with torch.inference_mode():
    images = trainer.ema.ema_model.sample(batch_size=16)

images_np = images.cpu().numpy()

result_list = []
conf_list = []
correct = 0
for img in images_np:
    res, conf = image2sequence(img.mean(0))
    result_list.append(res)
    conf_list.append(conf)
    correct += isValid(res)

print(result_list)
print(conf_list)
print(correct / 16.0)
