# PaDi

PaDi is a PyTorch implementation for long-term multivariate time series forecasting. The repository contains the model implementation, data loaders, experiment pipeline, and scripts for reproducing experiments on common benchmark datasets such as ETT, Electricity, Traffic, and Weather.

## Method Overview

PaDi is a patch-based framework for multivariate long-term time series forecasting. It first decomposes each input patch into a base component and a deviation component, allowing smooth local states and fine-grained fluctuations to be modeled separately before patch embedding.

For future prediction, PaDi constructs learnable future patch queries and injects a compact base summary as temporal anchors. These base-aware queries retrieve useful historical deviation patterns through Base-Aware Cross-Attention.

To model cross-variable dependencies, PaDi introduces Latent Channel Attention, which exchanges information through learnable latent channel bases instead of dense pairwise channel interactions. Overall, PaDi combines patch-wise decomposition and compact latent channel interaction for more effective multivariate forecasting.

## Code Structure

- `models/`: model implementation
- `layers/`: main model components
- `data/`: dataset loading and preprocessing
- `exp/`: training and evaluation pipeline
- `scripts/`: scripts for reproducing experiments
- `utils/`: utility functions
- `run.py`: main entry for training and evaluation

## Installation

The code was tested with Python 3.10.18 and PyTorch 2.4.1+cu121 

Install dependencies by:

```bash
pip install -r requirements.txt
```


## Data Preparation

Place datasets under the `dataset/` directory. The expected structure is:

```text
dataset/
  ETT-small/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  electricity/
    electricity.csv
  traffic/
    traffic.csv
  weather/
    weather.csv
```

For CSV datasets, the first column should be `date`, followed by feature columns. For multivariate forecasting, use `--features M`. The dataset path is controlled by `--root_path` and `--data_path` in `run.py` or the scripts under `scripts/`.

## Training

Use the provided scripts to reproduce the main experiments:

```bash
bash scripts/ETTh1_96_input.sh
bash scripts/ETTh2_96_input.sh
bash scripts/ETTm1_96_input.sh
bash scripts/ETTm2_96_input.sh
bash scripts/Electricity_96_input.sh
bash scripts/Traffic_96_input.sh
bash scripts/Weather_96_input.sh
```


## Outputs

Training and evaluation outputs are saved to:

- `checkpoints/`: model checkpoints, including `checkpoint.pth`
- `results/`: evaluation result folders for each experiment setting
- `result.txt`: appended summary metrics such as MSE, MAE
