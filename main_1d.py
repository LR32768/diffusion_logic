#from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch import Unet1D, MLP1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import argparse
import os

import torch
import random


def generate_dataset(d, n, eps, num_list=None):
    dataset = []
    if num_list is None:
        num_list = [d // 2 - 1, d // 2 + 1]
    all_seq = []
    # Generate all possible sequences with number of ones in num_list
    # Enumerate all binary sequences with d digits, check if the number of ones is in num_list
    # If yes, add to all_seq
    for i in range(2 ** d):
        seq = bin(i)[2:]
        seq = '0' * (d - len(seq)) + seq
        if seq.count('1') in num_list:
            all_seq.append(seq)
    print(len(all_seq))
    # Randomly choose n sequences from all_seq
    seqs = random.sample(all_seq, n)

    # Translate the sequences to tensors, add noise of eps standard deviation
    for seq in seqs:
        seq = torch.tensor([int(i) for i in seq]).float()
        dataset.append(seq + torch.randn_like(seq) * eps)
    # print(dataset)
    return torch.stack(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=8e-5, type=float, help='learning rate')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--length', type=int, default=16, help='input dimension')
    parser.add_argument('--datatype', type=str, default='plane')
    parser.add_argument('--size', type=int, default=512, help='dataset size')
    parser.add_argument('--eps', type=float, default=1e-2, help='noise level')
    parser.add_argument('--seed', type=int, default=None, help='seed for generating the dataset')

    # Model settings
    parser.add_argument('--model', type=str, default='mlp', help='model type (unet, mlp)')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--attn_blocks', type=int, default=1)
    parser.add_argument('--T', type=int, default=100)

    # Training settings
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num_steps', type=int, default=700000, help='number of steps to train')
    parser.add_argument('--see_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--stat_every', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ana_tgt', action='store_true', help='whether to use analytic target')

    args = parser.parse_args()

    num_list = [2*i for i in range(0, args.length // 2+1)] if args.datatype == 'parity' else None

    if os.path.exists(args.name):
        data = torch.load(os.path.join(args.name, 'dataset.pt'))
    else:
        data = generate_dataset(d=args.length, n=args.size, eps=args.eps, num_list=num_list)
    # save the dataset to args.name folder
        os.makedirs(args.name, exist_ok=True)
        torch.save(data, os.path.join(args.name, 'dataset.pt'))

    dataset = Dataset1D(data.unsqueeze(1))

    if args.model == 'unet':
        model = Unet1D(
            channels = 1,
            dim = args.width,
            dim_mults = (1, 2, 4, 8),
            attn_mid_num = args.attn_blocks,
        )
    elif args.model == 'mlp':
        model = MLP1D(
            dim = args.length,
            num_blocks = args.depth,
            dim_factor = max(1, args.width // args.length),
            time_emb_dim = args.width,
        )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.length,
        timesteps = args.T,            # number of steps
        sampling_timesteps = args.T,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        dataset_tensor = data,
        analytic_tgt = args.ana_tgt,
    )

    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size = args.bs,
        results_folder = args.name,
        train_lr = args.lr,
        train_num_steps = args.num_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        see_every = args.see_every,       # visualize progress every X steps
        save_every = args.save_every,     # save model every X steps
    )

    if args.resume is not None:
        trainer.load(args.resume)

    trainer.train()