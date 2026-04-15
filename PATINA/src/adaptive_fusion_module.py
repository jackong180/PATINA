import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_mask_like(mask, x):
    if mask is None:
        return None
    if mask.shape[-2:] == x.shape[-2:]:
        return mask
    return F.interpolate(mask, size=x.shape[-2:], mode='nearest')


def build_mask_context(mask, kernel_size=5):
    if mask is None:
        return None
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    pooled = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return torch.clamp(torch.maximum(mask, pooled), 0.0, 1.0)


class MaskConditionedGate(nn.Module):
    def __init__(self, channels, hidden_ratio=4):
        super().__init__()
        hidden = max(channels // hidden_ratio, 16)
        spatial_hidden = max(hidden // 2, 8)

        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, spatial_hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(spatial_hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature, mask):
        mask = resize_mask_like(mask, feature)
        if mask is None:
            return torch.ones_like(feature)

        pooled_feature = F.adaptive_avg_pool2d(feature, 1)
        pooled_mask = F.adaptive_avg_pool2d(mask, 1)
        channel = self.channel_gate(torch.cat((pooled_feature, pooled_mask), dim=1))

        mean_map = feature.mean(dim=1, keepdim=True)
        max_map = feature.abs().amax(dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat((mean_map, max_map, mask), dim=1))
        return channel * spatial


class LatentBranchMixer(nn.Module):
    def __init__(self, channels, num_branches=2, hidden_ratio=4, temperature_init=1.0):
        super().__init__()
        self.num_branches = int(num_branches)
        hidden = max(channels // hidden_ratio, 16)
        self.logit_head = nn.Sequential(
            nn.Conv2d(channels + 1 + self.num_branches, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_branches, kernel_size=1, bias=True),
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))

    def forward(self, x, branch_deltas, mask):
        if not branch_deltas:
            return torch.zeros_like(x)

        if len(branch_deltas) == 1:
            return branch_deltas[0]

        mask = resize_mask_like(mask, x)
        pooled_feature = F.adaptive_avg_pool2d(x, 1)
        pooled_mask = F.adaptive_avg_pool2d(mask, 1) if mask is not None else x.new_zeros((x.shape[0], 1, 1, 1))
        branch_energy = [
            F.adaptive_avg_pool2d(delta.abs().mean(dim=1, keepdim=True), 1)
            for delta in branch_deltas
        ]
        logits_input = torch.cat([pooled_feature, pooled_mask] + branch_energy, dim=1)
        logits = self.logit_head(logits_input)
        temperature = torch.clamp(self.temperature, min=0.25, max=4.0)
        weights = torch.softmax(logits / temperature, dim=1)

        mixed = torch.zeros_like(x)
        for index, delta in enumerate(branch_deltas):
            mixed = mixed + weights[:, index:index + 1] * delta
        return mixed


class MaskGuidedSkipFusion(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels, hidden_ratio=2):
        super().__init__()
        hidden = max(out_channels // hidden_ratio, 32)
        self.decoder_channels = int(decoder_channels)
        self.encoder_channels = int(encoder_channels)
        self.router = nn.Sequential(
            nn.Conv2d(self.decoder_channels + self.encoder_channels + 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, self.decoder_channels + self.encoder_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(self.decoder_channels + self.encoder_channels, out_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, decoder_feat, encoder_feat, mask):
        mask = resize_mask_like(mask, decoder_feat)
        boundary = torch.clamp(build_mask_context(mask, kernel_size=5) - mask, 0.0, 1.0)
        route = self.router(torch.cat((decoder_feat, encoder_feat, mask, boundary), dim=1))
        decoder_route, encoder_route = torch.split(
            route,
            [self.decoder_channels, self.encoder_channels],
            dim=1,
        )

        # Preserve strong valid-region skips while letting decoder features take
        # over inside holes and boundary transitions.
        decoder_weight = mask + (1.0 - mask) * (0.35 + 0.65 * decoder_route) + 0.50 * boundary
        encoder_weight = (1.0 - mask) + boundary * (0.35 + 0.65 * encoder_route)

        fused = torch.cat(
            (
                decoder_feat * decoder_weight,
                encoder_feat * encoder_weight,
            ),
            dim=1,
        )
        return self.project(fused)


class PretrainedSkipResidualAdapter(nn.Module):
    def __init__(self, decoder_channels, encoder_channels, out_channels, residual_scale_init=0.0, hidden_ratio=2):
        super().__init__()
        hidden = max(out_channels // hidden_ratio, 32)
        self.delta_proj = nn.Sequential(
            nn.Conv2d(
                decoder_channels + encoder_channels + out_channels + 2,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.gate = MaskConditionedGate(out_channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, decoder_feat, encoder_feat, base, mask):
        mask = resize_mask_like(mask, base)
        boundary = torch.clamp(build_mask_context(mask, kernel_size=5) - mask, 0.0, 1.0)
        delta = self.delta_proj(torch.cat((decoder_feat, encoder_feat, base, mask, boundary), dim=1))
        gate = self.gate(delta, build_mask_context(mask, kernel_size=5))
        return base + torch.tanh(self.residual_scale) * gate * delta


class RefinementResidualHead(nn.Module):
    def __init__(self, channels, bottleneck_channels, residual_scale_init=0.0, hidden_ratio=2):
        super().__init__()
        hidden = max(channels // hidden_ratio, 32)
        self.delta_proj = nn.Sequential(
            nn.Conv2d(channels + bottleneck_channels + 2, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.gate = MaskConditionedGate(channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, feature, bottleneck, mask):
        mask = resize_mask_like(mask, feature)
        boundary = torch.clamp(build_mask_context(mask, kernel_size=5) - mask, 0.0, 1.0)
        if bottleneck.shape[-2:] != feature.shape[-2:]:
            bottleneck = F.interpolate(bottleneck, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        delta = self.delta_proj(torch.cat((feature, bottleneck, mask, boundary), dim=1))
        gate = self.gate(delta, build_mask_context(mask, kernel_size=5))
        return feature + torch.tanh(self.residual_scale) * gate * delta
