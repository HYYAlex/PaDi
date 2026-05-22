import torch.nn as nn
from torch import Tensor

from layers.PaDi.backbone import Model_backbone


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        c_in = args.dec_in
        seq_len = args.seq_len
        self.pred_len = args.pred_len
        n_layers = args.d_layers
        n_heads = args.n_heads
        d_model = args.d_model
        d_ff = args.d_ff
        dropout = args.dropout
        independence = args.query_independence
        patch_len = args.patch_len
        stride = args.stride
        padding_patch = args.padding_patch
        store_attn = args.store_attn
        init = args.init

        channel_m = args.channel_m
        channel_drop = args.channel_drop
        channel_tau = args.channel_tau

        self.model = Model_backbone(
            c_in=c_in,
            seq_len=seq_len,
            pred_len=self.pred_len,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            independence=independence,
            store_attn=store_attn,
            padding_patch=padding_patch,
            init=init,
            channel_m=channel_m,
            channel_drop=channel_drop,
            channel_tau=channel_tau,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x[:, :self.pred_len, :]
