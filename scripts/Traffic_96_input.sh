export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

model_name=PaDi



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 862 \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --batch_size 32 \
  --train_epochs 30 \
  --channel_m 4 \
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 862 \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --batch_size 32 \
  --train_epochs 30 \
  --use_multi_gpu \
  --devices 0,1 \
  --channel_m 4 \
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 862 \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 32 \
  --lradj TST \
  --batch_size 32 \
  --train_epochs 30 \
  --channel_m 4 \
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 862 \
  --itr 1 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 32 \
  --patch_len 24 \
  --stride 24 \
  --batch_size 16 \
  --train_epochs 30 \
  --learning_rate 0.003 \
  --channel_m 4 \
