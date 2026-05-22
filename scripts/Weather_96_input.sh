export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4

model_name=PaDi

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 96 \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 32 \
  --train_epochs 10 \
  --batch_size 64 \
  --dropout 0.2 \
  --channel_m 10 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \




python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192\
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 192 \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 32 \
  --train_epochs 10 \
  --batch_size 64 \
  --dropout 0.2 \
  --channel_m 8 \
  --channel_drop 0.6 \
  --channel_tau 1.0 \



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 336 \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 32 \
  --train_epochs 10 \
  --batch_size 64 \
  --dropout 0.2 \
  --channel_m 8 \
  --channel_drop 0.6 \
  --channel_tau 1.0 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --init 0.0 \
  --seq_len 96 \
  --pred_len 720 \
  --d_layers 3 \
  --dec_in 21 \
  --itr 1 \
  --d_model 256 \
  --d_ff 256 \
  --n_heads 32 \
  --train_epochs 10 \
  --batch_size 64 \
  --dropout 0.4 \
  --channel_m 10 \
  --channel_drop 0.4 \
  --channel_tau 1.0 \
