export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

model_name=PaDi


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --pct_start 0.4 \
  --train_epochs 20 \
  --batch_size 32 \
  --channel_m 32 \
  --dropout 0.3 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --pct_start 0.4 \
  --train_epochs 20 \
  --batch_size 32 \
  --channel_m 32 \
  --dropout 0.3 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --pct_start 0.4 \
  --train_epochs 20 \
  --batch_size 32 \
  --channel_m 32 \
  --dropout 0.3 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 321 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --pct_start 0.4 \
  --train_epochs 20 \
  --batch_size 32 \
  --channel_m 32 \
  --dropout 0.3 \
