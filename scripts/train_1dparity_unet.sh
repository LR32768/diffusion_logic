srun -J parity -N 1 -p RTX3090 --gres gpu:1 python main_1d.py --model unet --bs 32 --see_every 500 --save_every 5000 --name unet_1d_parity