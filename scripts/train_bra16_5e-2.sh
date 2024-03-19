while (1<2)
do
  CUDA_VISIBLE_DEVICES=8 python main_bra.py --ratio 0.05 --name bra16_frac0.05 --bs 16 --resume latest
done