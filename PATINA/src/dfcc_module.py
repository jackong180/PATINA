import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_fusion_module import MaskConditionedGate, build_mask_context, resize_mask_like


class FourierUnitModified(nn.Module):
    """Embedded and adapted from Unbiased-Fast-Fourier-Convolution (ICCV 2023).

    This variant keeps the original frequency-mixing idea while making the
    spatial prior size configurable for PATINA stages.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        input_shape,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        ffc3d=False,
        fft_norm="ortho",
    ):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.input_shape = int(input_shape)
        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.loc_map = nn.Parameter(
            torch.rand(1, 1, self.input_shape, self.input_shape // 2 + 1)
        )
        self.lambda_base = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.conv_layer_down55 = nn.Conv2d(
            in_channels=in_channels * 2 + 1,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.groups,
            bias=False,
        )
        self.conv_layer_down55_shift = nn.Conv2d(
            in_channels=in_channels * 2 + 1,
            out_channels=out_channels * 2,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=self.groups,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

    def _resize_loc_map(self, target_h, target_w):
        if self.loc_map.shape[-2:] == (target_h, target_w):
            return self.loc_map
        return F.interpolate(
            self.loc_map,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        residual = x
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )
        else:
            orig_size = None

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1) + ffted.size()[3:])

        loc_map = self._resize_loc_map(ffted.shape[-2], ffted.shape[-1]).expand_as(
            ffted[:, :1, :, :]
        )
        ffted_copy = ffted.clone()

        cat_img_mask_freq = torch.cat(
            (
                ffted[:, : self.in_channels, :, :],
                ffted[:, self.in_channels :, :, :],
                loc_map,
            ),
            dim=1,
        )

        ffted = self.conv_layer_down55(cat_img_mask_freq)
        ffted = torch.fft.fftshift(ffted, dim=-2)
        ffted = self.relu(ffted)

        loc_map_shift = torch.fft.fftshift(loc_map, dim=-2)
        cat_img_mask_freq_shift = torch.cat(
            (
                ffted[:, : self.in_channels, :, :],
                ffted[:, self.in_channels :, :, :],
                loc_map_shift,
            ),
            dim=1,
        )

        ffted = self.conv_layer_down55_shift(cat_img_mask_freq_shift)
        ffted = torch.fft.ifftshift(ffted, dim=-2)

        lambda_base = torch.sigmoid(self.lambda_base)
        ffted = ffted_copy * lambda_base + ffted * (1.0 - lambda_base)

        ffted = ffted.view((batch, -1, 2) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2
        ).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if orig_size is not None:
            output = F.interpolate(
                output,
                size=orig_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        epsilon = 0.5
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = output - torch.mean(output) + torch.mean(x)
        min_value = x.detach().amin() - epsilon
        max_value = x.detach().amax() + epsilon
        output = torch.maximum(torch.minimum(output, max_value), min_value)
        output = torch.where(torch.isfinite(output), output, x)

        if output.dtype != residual.dtype:
            output = output.to(residual.dtype)
        return output


class DFCCResidualBlock(nn.Module):
    """Mask-aware corrective DFCC adapter for PATINA.

    The donor frequency block now predicts a gated correction `(freq - x)`
    instead of contributing the whole transformed feature map. This keeps the
    frequency branch compatible with stronger LCBC/MRDA defaults.
    """

    def __init__(
        self,
        channels,
        input_shape,
        groups=1,
        residual_scale_init=0.35,
        fft_norm="ortho",
        mask_context_kernel=5,
    ):
        super().__init__()
        self.fourier = FourierUnitModified(
            in_channels=channels,
            out_channels=channels,
            input_shape=input_shape,
            groups=groups,
            fft_norm=fft_norm,
        )
        self.pre_norm = nn.GroupNorm(1, channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.gate = MaskConditionedGate(channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.mask_context_kernel = int(mask_context_kernel)

    def forward(self, x, mask=None, return_delta=False):
        freq = self.fourier(self.pre_norm(x))
        corrective = self.refine(freq - x)
        if mask is None:
            mask_context = x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        else:
            mask_context = resize_mask_like(mask, x)
            mask_context = build_mask_context(mask_context, kernel_size=self.mask_context_kernel)

        gate = self.gate(corrective, mask_context)
        delta = torch.tanh(self.residual_scale) * gate * corrective
        if return_delta:
            return delta
        return x + delta
