import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class Projection(nn.Module):
    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.linear = nn.Linear(d_model, patch_len)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.flatten(x)
        return x
