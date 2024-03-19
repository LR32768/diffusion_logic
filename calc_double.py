import numpy as np
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from bra_gen import generate_dataset
from bra_ocr import *
import torch
import os
import argparse

#generate_dataset('bra16', 16, 8)
def get_samples(trainer, num=128, bs=16):
    images = None
    with torch.inference_mode():
        for _ in range(num // bs):
            img = trainer.ema.ema_model.sample(batch_size=bs)
            img = img.cpu().numpy()

            if images is not None:
                images = np.concatenate((images, img), 0)
            else:
                images = img
    return images.mean(1)

# def load_dataset_seq(datapath='./bra16'):
#     dataset_list = []
#     for root, ds, fs in os.walk(datapath):
#         for f in fs:
#             result = ocr(os.path.join(datapath, f), templates)
#             dataset_list.append(result)
#     return dataset_list

def edit_distance(s):
    L = len(s)
    m = L // 2
    n = L - m
    s1 = s[:m]
    s2 = s[m:]
    # Compute the edit distance between s1 and s2, note that both s1 and s2 could be empty
    dp = np.zeros((m+1, n+1), dtype=np.int)
    for i in range(m+1):
        dp[i, 0] = i
    for j in range(n+1):
        dp[0, j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + 1
    return dp[m, n]


def stat_model(trainer, epoch, num=128, bs=16):
    trainer.load(epoch)
    trainer.ema.ema_model.eval()
    images_np = get_samples(trainer, num, bs=bs)

    dist_sum = 0
    exact_ratio = 0
    res_list = []
    dist_list = []
    for img in images_np:
        #res, conf = image2sequence(img)
        res = ocr(img, templates, mode='np')
        res_list.append(res)
        if not '?' in res and len(res) > 0:
            dist_list.append(edit_distance(res))
            if edit_distance(res) == 0:
                exact_ratio += 1
        else:
            dist_list.append(8)

    print(res_list)
    return np.array(dist_list).mean(), np.array(dist_list).std(), exact_ratio / len(res_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--name', '-n', type = str, default='exp', help='name of the experiment')

    # Dataset hyperparameters
    parser.add_argument('--size', type=int, default=256, help='input sequence length')
    parser.add_argument('--up', type=int, default=60)
    parser.add_argument('--down', type=int, default=1)

    # Model settings
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--T', type=int, default=100)

    # Training settings
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--num', type=int, default=128)
    args = parser.parse_args()


    model = Unet(
        dim = args.width,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = args.size,
        timesteps = args.T,           # number of steps
        sampling_timesteps = args.T    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    trainer = Trainer(
        diffusion,
        f'/cluster/home1/lurui/ddpm_bra/{args.name}_data',
        train_batch_size = args.bs,
        results_folder = args.name,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False             # whether to calculate fid during training
    )

    templates = {
            '(': preprocess_image('templates/template(.png'),
            ')': preprocess_image('templates/template).png'),
            '[': preprocess_image('templates/template[.png'),
            ']': preprocess_image('templates/template].png'),
            '{': preprocess_image('templates/template{.png'),
            '}': preprocess_image('templates/template}.png')
        }

    # dataset_list = load_dataset_seq(f'/cluster/home1/lurui/ddpm_bra/{args.name}_data')
    # print(dataset_list)


    # r1, r2, r3, r4 = stat_model(trainer, args.up, 128)
    # print(f"{r1} {r2} {r3} {r4}")
    #
    for i in range(args.down, args.up):
        print(f"Compute {i*5} epoch ...")
        avg, std, exact = stat_model(trainer, i*5, args.num, args.bs)
        print(f"avg:{avg}, std:{std}, exact ratio {exact}")
        with open(f"stat_{args.name}.txt", 'a') as f:
            f.write(f"{avg} {std} {exact}\n")

    # with open(f"stat_{args.name}.txt", 'a') as f:
    #     f.write(f"{r1} {r2} {r3} {r4}\n")