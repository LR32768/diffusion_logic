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

def load_dataset_seq(datapath='./bra16'):
    dataset_list = []
    for root, ds, fs in os.walk(datapath):
        for f in fs:
            result = ocr(os.path.join(datapath, f), templates)
            dataset_list.append(result)
    return dataset_list

def stat_model(trainer, epoch, num=128):
    trainer.load(epoch)
    trainer.ema.ema_model.eval()
    images_np = get_samples(trainer, num)

    N1 = 0 # Not even bracket
    N2 = 0 # Bracket but not balanced
    N3 = 0 # Balanced but in dataset
    N4 = 0 # Balanced and novel
    res_list = []
    for img in images_np:
        #res, conf = image2sequence(img)
        res = ocr(img, templates, mode='np')
        res_list.append(res)
        if '?' in res or len(res) == 0:
            N1 += 1
        elif not isValid(res):
            N2 += 1
        elif res in dataset_list:
            N3 += 1
        else:
            N4 += 1
    print(res_list)
    return N1 / num, N2 / num, N3 / num, N4 / num


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
    parser.add_argument('--bs', type=int, default=8, help='batch size')

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

    dataset_list = load_dataset_seq(f'/cluster/home1/lurui/ddpm_bra/{args.name}_data')
    print(dataset_list)

    for i in range(args.down, args.up):
        print(f"Compute {i*5} epoch ...")
        r1, r2, r3, r4 = stat_model(trainer, i*5, 128)
        print(f"{r1} {r2} {r3} {r4}")
        with open(f"stat_{args.name}.txt", 'a') as f:
            f.write(f"{r1} {r2} {r3} {r4}\n")