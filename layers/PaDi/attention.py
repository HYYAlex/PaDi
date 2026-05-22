from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseAwareCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.scale = self.d_h ** -0.5

        self.W_Q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_bias: Optional[Tensor] = None,
        value_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        bs = Q.size(0)

        q = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_h)
        k = self.W_K(K).view(bs, -1, self.n_heads, self.d_h)
        v = self.W_V(V).view(bs, -1, self.n_heads, self.d_h)

        if value_bias is not None:
            if value_bias.dim() == 2:
                value_bias = value_bias.unsqueeze(0).expand(bs, -1, -1)
            elif value_bias.dim() == 3 and value_bias.size(0) == 1 and bs != 1:
                value_bias = value_bias.expand(bs, -1, -1)
            vb = self.W_V(value_bias).view(bs, -1, self.n_heads, self.d_h)
            v = v + vb

        scores = torch.einsum("bphd,bshd->bphs", q, k) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bphs,bshd->bphd", attn, v).contiguous().view(bs, -1, self.n_heads * self.d_h)
        out = self.to_out(out)
        return out, attn, scores


class LatentChannelAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_anchors: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        tau: float = 1.0,
        share_A_over_time: bool = True,
        pre_norm: bool = True,
        init_std: float = 0.02,
        store_stats: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_h = self.d_model // self.n_heads
        self.n_anchors = int(n_anchors)
        self.eps = float(eps)
        self.tau = float(tau)
        self.share_A_over_time = bool(share_A_over_time)
        self.pre_norm = bool(pre_norm)
        self.store_stats = bool(store_stats)

        self.scale = self.d_h ** -0.5

        self.W_K = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=True)
        self.W_O = nn.Linear(self.d_model, self.d_model, bias=True)

        self.anchors = nn.Parameter(torch.randn(self.n_heads, self.n_anchors, self.d_h) * float(init_std))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x: Tensor) -> Tensor:
        B, C, S, D = x.shape
        assert D == self.d_model

        x0 = x
        if self.pre_norm:
            x = self.norm(x)

        v = self.W_V(x).view(B, C, S, self.n_heads, self.d_h)

        if self.share_A_over_time:
            x_pool = x.mean(dim=2)
            k_pool = self.W_K(x_pool).view(B, C, self.n_heads, self.d_h)
            scores = torch.einsum("bchd,hmd->bchm", k_pool, self.anchors) * (self.scale / max(self.tau, 1e-6))
            A = torch.softmax(scores, dim=-1)
            delta = A.sum(dim=1) + self.eps
            M1 = torch.einsum("bchm,bcshd->bshmd", A, v)
            M2 = M1 / delta.unsqueeze(1).unsqueeze(-1)
            out = torch.einsum("bchm,bshmd->bcshd", A, M2)
        else:
            k = self.W_K(x).view(B, C, S, self.n_heads, self.d_h)
            scores = torch.einsum("bcshd,hmd->bcshm", k, self.anchors) * (self.scale / max(self.tau, 1e-6))
            A = torch.softmax(scores, dim=-1)
            delta = A.sum(dim=1) + self.eps
            M1 = torch.einsum("bcshm,bcshd->bshmd", A, v)
            M2 = M1 / delta.unsqueeze(-1)
            out = torch.einsum("bcshm,bshmd->bcshd", A, M2)

        out = out.contiguous().view(B, C, S, D)
        out = self.W_O(out)
        out = self.drop(out)

        g = torch.sigmoid(self.gate)
        y = x0 + g * out

        if self.store_stats:
            with torch.no_grad():
                if self.share_A_over_time:
                    self.A_mean = A.mean(dim=2)
                    self.delta_mean = delta.mean(dim=1)
                else:
                    self.A_mean = A.mean(dim=(2, 3))
                    self.delta_mean = delta.mean(dim=2).mean(dim=1)

        return y
