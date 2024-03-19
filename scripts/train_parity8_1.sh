srun -J par8_1.0 -N 1 -p RTX3090 --gres gpu:1  python main_bra.py --ratio 0.5 --type parity --name parity8_frac1.0 --bs 16 --num_para 8 --resume latest
