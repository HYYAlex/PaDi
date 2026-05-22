from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

class LocalLinearDetrender(nn.Module):

    def __init__(self, patch_len: int):
        super().__init__()
        t = torch.arange(float(patch_len), dtype=torch.float32)
        t_centered = t - t.mean()
        denom = (t_centered ** 2).sum().clamp_min(1e-6)
        self.register_buffer("t_centered", t_centered.view(1, 1, 1, patch_len), persistent=False)
        self.register_buffer("denom", denom.view(1, 1, 1, 1), persistent=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        slope = (x * self.t_centered).sum(dim=-1, keepdim=True) / self.denom
        trend = mean + slope * self.t_centered
        deviation = x - trend
        return trend, deviation

class BaseFutureInjector(nn.Module):

    def __init__(self, d_model: int, pred_patch_num: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(64, d_model // 2)
        if pred_patch_num > 1:
            horizon = torch.linspace(0.0, 1.0, steps=pred_patch_num, dtype=torch.float32).view(1, pred_patch_num, 1)
        else:
            horizon = torch.zeros(1, pred_patch_num, 1, dtype=torch.float32)
        self.register_buffer("inject_horizon", horizon, persistent=False)

        self.inject_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.inject_gate = nn.Sequential(
            nn.Linear(2 * d_model + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.inject_gate[-1].weight)
        self.inject_gate[-1].bias.data.fill_(0.0)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)
        self.last_gate = None

    def forward(self, pred: Tensor, base_mem: Tensor) -> Tensor:
        base_summary = base_mem.mean(dim=1, keepdim=True).expand(-1, pred.size(1), -1)
        horizon = self.inject_horizon
        if horizon.size(0) != pred.size(0):
            horizon = horizon.expand(pred.size(0), -1, -1)
        inject_in = torch.cat([pred, base_summary, horizon], dim=-1)
        inject_delta = self.inject_mlp(inject_in)
        inject_gate = torch.sigmoid(self.inject_gate(inject_in))
        self.last_gate = inject_gate.detach()
        pred = pred + self.out_drop(inject_gate * inject_delta)
        pred = self.out_norm(pred)
        return pred
