import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_fusion_module import MaskConditionedGate, resize_mask_like


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LCBCAdapter(nn.Module):
    """Mask-conditioned LCBC adapter for PATINA.

    Compared with the original CHF splice, this version treats contextual
    attention as a correction branch instead of injecting the full attended
    feature map directly. That lets LCBC stay expressive while reducing
    destructive interference with MRDA/DFCC on easy regions.
    """

    def __init__(
        self,
        channels,
        embed_dim=96,
        softmax_scale=10.0,
        residual_scale_init=0.10,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.softmax_scale = float(softmax_scale)

        self.norm = ChannelLayerNorm(channels)
        self.to_q = nn.Conv2d(channels, self.embed_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, self.embed_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.query_mask_proj = nn.Conv2d(1, self.embed_dim, kernel_size=1, bias=False)
        self.delta_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.gate = MaskConditionedGate(channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x, mask, return_delta=False):
        mask = resize_mask_like(mask, x)
        b, _, h, w = x.shape

        x_norm = self.norm(x)
        q = self.to_q(x_norm) + self.query_mask_proj(mask)
        q = q.flatten(2).transpose(1, 2)                  # B, N, D
        k = self.to_k(x_norm).flatten(2).transpose(1, 2)  # B, N, D
        v = self.to_v(x).flatten(2).transpose(1, 2)       # B, N, C

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(1, 2)) * self.softmax_scale / math.sqrt(self.embed_dim)

        valid_tokens = (mask.flatten(2).transpose(1, 2) < 0.5)
        attn = attn.masked_fill(~valid_tokens.transpose(1, 2), -1e4)
        attn = torch.softmax(attn, dim=-1)

        attended = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, h, w)
        attended = self.out_proj(attended)
        corrective = self.delta_proj(torch.cat((attended - x, x_norm * mask), dim=1))
        gate = self.gate(corrective, mask) * mask
        delta = torch.tanh(self.residual_scale) * gate * corrective

        if return_delta:
            return delta
        return x + delta
