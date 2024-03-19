srun -J parity -N 1 -p RTX3090 --gres gpu:1 python main_bra.py --ratio 2.13e-4 --name bra16_threepara_num2000 --bs 16 --type three --save_every 5000 --num_steps 1000000 --resume latest
