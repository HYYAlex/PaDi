from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from layers.PaDi.attention import BaseAwareCrossAttention
from layers.PaDi.layers import GEGLU


class Decoder(nn.Module):
    def __init__(
        self,
        seq_patch_num: int,
        d_model: int,
        n_heads: int,
        pred_patch_num: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        n_layers: int = 1,
        store_attn: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(
                seq_patch_num=seq_patch_num,
                d_model=d_model,
                pred_patch_num=pred_patch_num,
                n_heads=n_heads,
                d_ff=d_ff or (4 * d_model),
                store_attn=store_attn,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, seq_dev: Tensor, pred: Tensor, dev_value_bias: Optional[Tensor] = None) -> Tensor:
        for mod in self.layers:
            seq_dev, pred = mod(
                seq_dev=seq_dev,
                pred=pred,
                dev_value_bias=dev_value_bias,
            )
        return pred

class DecoderLayer(nn.Module):
    def __init__(
        self,
        seq_patch_num: int,
        d_model: int,
        pred_patch_num: int,
        n_heads: int,
        d_ff: int = 256,
        store_attn: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.deviation_attn = BaseAwareCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            proj_dropout=dropout,
        )

        self.norm_attn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_model, bias=bias),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.store_attn = bool(store_attn)


        self.attn_dev = None

    def forward(
        self,
        seq_dev: Tensor,
        pred: Tensor,
        dev_value_bias: Optional[Tensor] = None,
    ):
        upd_dev, attn_dev, scores_dev = self.deviation_attn(
            Q=pred,
            K=seq_dev,
            V=seq_dev,
            value_bias=dev_value_bias,
        )

        if self.store_attn:
            self.attn_dev = attn_dev.detach()

        pred = pred + upd_dev
        pred = self.norm_attn(pred)

        pred2 = self.ffn(pred)
        pred = pred + pred2
        pred = self.norm_ffn(pred)

        return seq_dev, pred
