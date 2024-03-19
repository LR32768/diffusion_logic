from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
parser.add_argument('--name', '-n', type=str, default='exp', help='name of the experiment')

# Dataset hyperparameters
parser.add_argument('--datapath', type=str, default='/cluster/home1/lurui/ffhq128', help='path to the dataset')
parser.add_argument('--img_size', type=int, default=64, help='dataset size')
parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')

# Model settings
parser.add_argument('--model', type=str, default='mlp', help='model type (unet, mlp)')
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--T', type=int, default=100)

# Training settings
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--num_steps', type=int, default=200000, help='number of steps to train')
parser.add_argument('--see_every', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=5000)
parser.add_argument('--stat_every', type=int, default=500)
parser.add_argument('--no_flip', action='store_true', help='disable horizontal flip augmentation
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--ana_tgt', action='store_true', help='whether to use analytic target')

args = parser.parse_args()

model = Unet(
    dim = args.width,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = args.img_size,
    timesteps = args.T,            # number of steps
    sampling_timesteps = args.T    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    args.datapath,
    train_batch_size = args.bs,
    results_folder = args.name,
    train_lr = args.lr,
    train_num_steps = args.num_steps, # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,            # whether to calculate fid during training
    augment_horizontal_flip = not args.no_flip, # whether to use horizontal flip augmentation
)
if args.resume is not None:
    trainer.load_model(args.resume)
trainer.train()