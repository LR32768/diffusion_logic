#from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch import Unet1D, MLP1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import argparse
import os
import math
import torch
import random
from tqdm import tqdm

def cal_exact_score(data, alpha, x):
    _data = 2 * data - 1
    diff = _data * torch.sqrt(alpha) - x
    d2 = torch.sum(diff**2, axis=-1) / (2*(1-alpha.reshape(-1, 1)))
    mind2, _ = torch.min(d2, dim=1, keepdim=True)
    d2 -= mind2
    kernel = torch.exp(-d2)
    kernel /= torch.sum(kernel, dim=1, keepdim=True)

    weight_x0 = kernel @ _data
    weight_x0 = weight_x0.unsqueeze(1)

    score = x / torch.sqrt(1-alpha) - torch.sqrt(alpha / (1-alpha)) * weight_x0
    return score

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
    n = min(n, len(all_seq))
    seqs = random.sample(all_seq, n)

    # Translate the sequences to tensors, add noise of eps standard deviation
    for seq in seqs:
        seq = torch.tensor([int(i) for i in seq]).float()
        dataset.append(seq + torch.randn_like(seq) * eps)
    # print(dataset)
    return torch.stack(dataset)

def feature_plausibility(x, pred, real, feature='phi1'):
    diff = real - pred
    if feature == 'phi1': # whether all the entries are near -1 or 1
        nabla_phix = torch.sign(x)
    elif feature == 'phi2': # Whether total number of 1s is correct
        nabla_phix = torch.ones_like(x).to(x.device)
    elif feature == 'phi3': # Whether the parity is correct
        prod = torch.prod(x, dim=-1).unsqueeze(-1)
        nabla_phix = prod / x

    dot = torch.sum(diff * nabla_phix, dim=-1)
    # print(f"{feature} abs(dot):",torch.abs(dot).shape)
    # print(f"{feature} norm(nabla):",torch.norm(nabla_phix, dim=-1).shape)
    result = torch.abs(dot) / torch.norm(nabla_phix, dim=-1)
    return (result**2).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--length', type=int, default=16, help='input dimension')
    parser.add_argument('--datatype', type=str, default='plane')
    # Model settings
    parser.add_argument('--model', type=str, default='mlp', help='model type (unet, mlp)')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--attn_blocks', type=int, default=1)
    parser.add_argument('--T', type=int, default=100)

    parser.add_argument('--num_epochs', type=int, default=700)
    parser.add_argument('--st', type=int, default=1)
    parser.add_argument('--gap', type=int, default=10)

    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--fix_t', type=int, default=-1)

    # Training settings
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    args = parser.parse_args()


    # Generate full dataset
    num_list = [2 * i for i in range(0, args.length // 2 + 1)] if args.datatype == 'parity' else None
    total_num = 2**args.length
    data = generate_dataset(args.length, total_num, 1e-2, num_list)
    total_num = len(data)
    dataset = Dataset1D(data)

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
            # num_blocks = args.depth,
            dim_factor = max(1, args.width // args.length),
            time_emb_dim = args.width,
        )

    model = model.cuda()
    data = data.cuda()

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = args.length,
        timesteps = args.T,            # number of steps
        sampling_timesteps = args.T,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        dataset_tensor = data,
        # analytic_tgt = ,
    )

    trainer = Trainer1D(
        diffusion,
        dataset=dataset,
        train_batch_size=args.bs,
        results_folder=args.name,
        train_lr=1e-3,
        train_num_steps=1000,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        see_every=100,  # visualize progress every X steps
        save_every=100,  # save model every X steps
    )

    for epoch in range(args.st, args.num_epochs // args.gap + 1):
        # diffusion.load(epoch * args.gap)
        trainer.load(epoch * args.gap)
        avg_error = 0
        avg_cos = 0
        avg_p1 = 0
        avg_p2 = 0
        avg_p3 = 0
        for _ in tqdm(range(args.N)):
            batch = min(args.bs, total_num)
            # random sample batch within data to get feed_data
            indices = torch.randperm(len(dataset))[:batch]
            feed_data = data[indices]
            x, pred_score, alpha = trainer.forward_score(feed_data.unsqueeze(1), fix_t=args.fix_t)
            exact_score = cal_exact_score(data, alpha, x)
            avg_error += torch.mean(((pred_score - exact_score)**2).sum(-1)).item()
            avg_cos += torch.mean(torch.nn.functional.cosine_similarity(pred_score, exact_score)).item()
            avg_p1 += feature_plausibility(x, pred_score, exact_score, 'phi1').item()
            avg_p2 += feature_plausibility(x, pred_score, exact_score, 'phi2').item()
            avg_p3 += feature_plausibility(x, pred_score, exact_score, 'phi3').item()
        avg_error /= args.N
        avg_cos /= args.N
        avg_p1 /= args.N
        avg_p2 /= args.N
        avg_p3 /= args.N

        print(f'Epoch {epoch * args.gap}, avg_error: {avg_error}, avg_cos: {avg_cos}')
        print(f'avg_p1: {avg_p1}, avg_p2: {avg_p2}, avg_p3: {avg_p3}')
