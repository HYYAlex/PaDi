export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

model_name=PaDi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 128 \
  --train_epochs 15 \
  --dropout 0.3 \
  --channel_m 1 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --d_layers 2 \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --batch_size 128 \
  --train_epochs 15 \
  --dropout 0.4 \
  --channel_m 5 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --d_layers 2 \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --train_epochs 15 \
  --batch_size 128 \
  --dropout 0.4 \
  --channel_m 5 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 7 \
  --itr 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 32 \
  --train_epochs 15 \
  --batch_size 128 \
  --dropout 0.3 \
  --channel_m 1 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \
