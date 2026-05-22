import torch
import torch.nn as nn
from torch import Tensor

from layers.PaDi.embedding import Dummy_Embedding
from layers.PaDi.layers import Projection


class Model_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 24,
        stride: int = 24,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.0,
        independence: bool = False,
        store_attn: bool = False,
        padding_patch=None,
        init: float = -12.0,
        channel_m: int = 16,
        channel_drop: float = 0.0,
        channel_tau: float = 1.0,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        pred_patch_num = (pred_len + patch_len - 1) // patch_len
        seq_patch_num = int((seq_len - patch_len) / stride + 1)

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            seq_patch_num += 1

        self.backbone = Dummy_Embedding(
            c_in=c_in,
            seq_patch_num=seq_patch_num,
            patch_len=patch_len,
            pred_patch_num=pred_patch_num,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            independence=independence,
            store_attn=store_attn,
            init=init,
            use_channel=True,
            channel_m=channel_m,
            channel_drop=channel_drop,
            channel_tau=channel_tau,
        )

        self.n_vars = c_in
        self.pred_len = pred_len
        self.proj = Projection(d_model, patch_len)

    def forward(self, z: Tensor) -> Tensor:
        mean = z.mean(2, keepdim=True)
        std = torch.sqrt(torch.var(z, dim=2, keepdim=True, unbiased=False) + 1e-5)
        z = (z - mean) / std

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        z = self.backbone(z)
        z = self.proj(z)

        z = z[:, :, :self.pred_len]
        z = z * std[:, :, 0].unsqueeze(2)
        z = z + mean[:, :, 0].unsqueeze(2)
        return z
