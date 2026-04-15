import torch
import torch.nn as nn

from .adaptive_fusion_module import MaskConditionedGate, resize_mask_like


class MRDADownsampleAdapter(nn.Module):
    """Corrective MRDA branch for PATINA.

    The original donor branch used a full residual path, which
    forced the default scale to stay tiny. Here the branch predicts a
    mask-conditioned correction around the SEM downsample baseline.
    """

    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        if out_channels * 2 != in_channels * 4:
            raise ValueError(
                f"MRDA expects out_channels={in_channels * 2} for PATINA backbone compatibility, got in={in_channels}, out={out_channels}"
            )

        self.feature_body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.PixelUnshuffle(2),
        )
        self.mask_body = nn.PixelUnshuffle(2)
        mask_hidden = max(out_channels // 4, 8)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(4, mask_hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mask_hidden, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                bias=bias,
            ),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=bias),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.gate = MaskConditionedGate(out_channels)

    def forward(self, x, mask, base=None, return_delta=False):
        feat = self.feature_body(x)
        mask_unshuffled = self.mask_body(mask)
        mask_feat = self.mask_encoder(mask_unshuffled)
        mrda_feature = self.proj(torch.cat((feat, mask_feat), dim=1))
        mrda_feature = self.refine(mrda_feature)

        if base is None:
            base = feat

        corrective = mrda_feature - base
        mask_target = resize_mask_like(mask, corrective)
        gate = self.gate(corrective, mask_target)
        delta = gate * corrective
        if return_delta:
            return delta
        return base + delta
