import torch
import torch.nn as nn
from torch import Tensor

from layers.PaDi.attention import LatentChannelAttention
from layers.PaDi.decomposition import BaseFutureInjector, LocalLinearDetrender
from layers.PaDi.decoder import Decoder


class Dummy_Embedding(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_patch_num: int,
        patch_len: int,
        pred_patch_num: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.0,
        store_attn: bool = False,
        independence: bool = False,
        init: float = -12.0,
        use_channel: bool = False,
        channel_m: int = 16,
        channel_drop: float = 0.0,
        channel_tau: float = 1.0,
    ):
        super().__init__()

        self.c_in = c_in
        self.pred_patch_num = pred_patch_num
        self.patch_len = patch_len
        self.independence = bool(independence)
        self.store_attn = bool(store_attn)

        self.patch_encoder = nn.Linear(patch_len, d_model)
        self.base_value_encoder = nn.Linear(patch_len, d_model, bias=True)
        nn.init.zeros_(self.base_value_encoder.weight)
        nn.init.zeros_(self.base_value_encoder.bias)

        self.dropout = nn.Dropout(dropout)
        self.pred_ln = nn.LayerNorm(d_model)
        self.pred_drop = nn.Dropout(0.1)

        self.detrender = LocalLinearDetrender(patch_len)

        if self.independence:
            self.dummies = nn.Parameter(0.5 * torch.randn(pred_patch_num, patch_len))
        else:
            self.dummy_base = nn.Parameter(0.5 * torch.randn(pred_patch_num, patch_len))
            self.dummy_res = nn.Parameter(torch.zeros(c_in, pred_patch_num, patch_len))
            self.dummy_gate = nn.Parameter(torch.full((c_in, 1, 1), float(init)))

        self.PE_hist = nn.Parameter(0.04 * torch.rand(seq_patch_num, d_model) - 0.02)
        self.PE_pred = nn.Parameter(0.04 * torch.rand(pred_patch_num, d_model) - 0.02)

        self.use_channel = bool(use_channel)
        if self.use_channel:
            mixer_kwargs = dict(
                d_model=d_model,
                n_heads=n_heads,
                n_anchors=channel_m,
                dropout=channel_drop,
                eps=1e-6,
                tau=channel_tau,
                share_A_over_time=True,
                pre_norm=True,
                store_stats=False,
            )
            self.base_channel = LatentChannelAttention(**mixer_kwargs)
            self.deviation_channel = LatentChannelAttention(**mixer_kwargs)

            self.level_channel = self.base_channel

        self.base_injector = BaseFutureInjector(
            d_model=d_model,
            pred_patch_num=pred_patch_num,
            dropout=dropout,
        )
        self.base_inject_gate = None

        self.decoder = Decoder(
            seq_patch_num=seq_patch_num,
            d_model=d_model,
            n_heads=n_heads,
            pred_patch_num=pred_patch_num,
            d_ff=d_ff,
            dropout=dropout,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def _build_prediction_patches(self, bs: int, n_vars: int) -> Tensor:
        if self.independence:
            pred_patch = self.dummies.unsqueeze(0).unsqueeze(0).repeat(bs, n_vars, 1, 1)
        else:
            gate = torch.sigmoid(self.dummy_gate)
            pred_patch = self.dummy_base.unsqueeze(0) + gate * self.dummy_res
            pred_patch = pred_patch.unsqueeze(0).repeat(bs, 1, 1, 1)
        return pred_patch

    def forward(self, x: Tensor) -> Tensor:

        bs, n_vars, seq_patch_num, _ = x.shape

        raw_patch = x
        base_patch, deviation_patch = self.detrender(raw_patch)

        base_mem = self.patch_encoder(base_patch) + self.PE_hist
        deviation_mem = self.patch_encoder(deviation_patch) + self.PE_hist
        deviation_value_bias = self.base_value_encoder(base_patch)

        if self.use_channel:
            base_mem = self.base_channel(base_mem)
            deviation_mem = self.deviation_channel(deviation_mem)

        pred_patch = self._build_prediction_patches(bs=bs, n_vars=n_vars)
        pred = self.patch_encoder(pred_patch) + self.PE_pred
        pred = self.pred_drop(self.pred_ln(pred))

        seq_base = base_mem.reshape(bs * n_vars, base_mem.shape[2], base_mem.shape[3])
        seq_dev = deviation_mem.reshape(bs * n_vars, deviation_mem.shape[2], deviation_mem.shape[3])
        base_bias = deviation_value_bias.reshape(bs * n_vars, deviation_value_bias.shape[2], deviation_value_bias.shape[3])
        pred = pred.reshape(bs * n_vars, pred.shape[2], pred.shape[3])

        if self.training and self.dropout.p > 0:
            keep_prob = 1.0 - self.dropout.p
            drop_mask = (torch.rand_like(seq_dev) < keep_prob).to(seq_dev.dtype) / keep_prob
            seq_base = seq_base * drop_mask
            seq_dev = seq_dev * drop_mask

        pred = self.base_injector(pred=pred, base_mem=seq_base)
        if self.store_attn:
            self.base_inject_gate = self.base_injector.last_gate

        z = self.decoder(seq_dev=seq_dev, pred=pred, dev_value_bias=base_bias)
        z = z.reshape(bs, n_vars, z.shape[-2], z.shape[-1])
        return z
