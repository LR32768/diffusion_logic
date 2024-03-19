from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from bra_gen import generate_dataset
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--num_para', type=int, default=16, help='number of hills in dataset')
    parser.add_argument('--depth', type=int, default=8, help='manifold dimensionality')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--size', type=int, default=256, help='input sequence length')
    parser.add_argument('--type', type=str, default='one')
    parser.add_argument('--font', type=str, default='./times.ttf')
    parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')

    # Model settings
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--T', type=int, default=100)
    parser.add_argument('--attn_blocks', type=int, default=1, help='number of attention layers')

    # Training settings
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_steps', type=int, default=700000, help='number of steps to train')
    parser.add_argument('--see_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--stat_every', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    if not os.path.exists(f'{args.name}_data'):
        generate_dataset(f'{args.name}_data', length=args.num_para, stack_depth=args.depth,
                         frac=args.ratio, datatype=args.type, font=args.font)

    model = Unet(
        dim = args.width,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False,
        attn_blocks = args.attn_blocks,
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.size,
        timesteps = args.T,            # number of steps
        sampling_timesteps = args.T    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        f'{args.name}_data',
        train_batch_size = args.bs,
        results_folder = args.name,
        train_lr = args.lr,
        train_num_steps = args.num_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False             # whether to calculate fid during training
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.train()