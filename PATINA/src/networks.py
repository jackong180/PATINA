import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint
from .adaptive_fusion_module import (
    LatentBranchMixer,
    MaskGuidedSkipFusion,
    PretrainedSkipResidualAdapter,
    RefinementResidualHead,
    build_mask_context,
    resize_mask_like,
)
from .dfcc_module import DFCCResidualBlock
from .lcbc_module import LCBCAdapter
from .mrda_module import MRDADownsampleAdapter

import math



from pdb import set_trace as stx
import numbers
from einops import rearrange


def _disabled_cuda_autocast():
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', enabled=False)
    return torch.cuda.amp.autocast(enabled=False)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

############# Restormer-inpainting
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Spatially Enhanced Feed-Forward Network (SEFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)    


        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3,stride=1,padding=1,bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)
        

    def forward(self, x, spatial):
        
        
        x = self.project_in(x)
        
        
        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)  
        ####
        

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y),dim=1))
        x1 = self.dwconv_afterfusion(x1)
        
        
        
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class PreConditionBlock(nn.Module):
    """Lightweight mask-conditioned pre-modulation block.

    This replaces a second full FFN at high resolution to keep the extra
    collaboration path affordable during training.
    """

    def __init__(self, dim, bias):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, cond):
        y = self.in_proj(x + cond)
        y = F.gelu(self.dwconv(y))
        y = self.out_proj(y)
        return y


class RefinementLite(nn.Module):
    """Lightweight full-resolution refinement head."""

    def __init__(self, channels, num_blocks=4, bias=False):
        super().__init__()
        self.mask_adapter = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=bias),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias),
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask):
        mask = resize_mask_like(mask, x)
        boundary = torch.clamp(build_mask_context(mask, kernel_size=5) - mask, 0.0, 1.0)
        cond = self.mask_adapter(torch.cat((mask, boundary), dim=1))
        for block in self.blocks:
            x = x + block(x + cond)
        return x




##########################################################################
## Snake Bi-Directional Sequence Modelling (SBSM)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 enable_mask_precondition=False, enable_mask_route=False,
                 precondition_scale_init=0.0, route_scale_init=0.0):
        super(TransformerBlock, self).__init__()

        self.enable_mask_precondition = bool(enable_mask_precondition) and dim >= 192
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias)
        if self.enable_mask_precondition:
            self.mask_adapter = nn.Sequential(
                nn.Conv2d(2, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            )
            self.pre_scale = nn.Parameter(torch.tensor(float(precondition_scale_init)))
        else:
            self.mask_adapter = None
            self.pre_scale = None
        self.use_gradient_checkpoint = True
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)


        
        ##### Try Mamba
        self.attn = MambaLayer(
            dim,
            enable_mask_route=enable_mask_route,
            route_scale_init=route_scale_init,
        )
        #####

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x, pos, mask = x[0], x[1], x[2]
        mask = resize_mask_like(mask, x)
        mask_context = build_mask_context(mask, kernel_size=5)
        x_spatial = x
        if self.enable_mask_precondition:
            boundary = torch.clamp(mask_context - mask, 0.0, 1.0)
            norm1_1_x = self.norm1_1(x)
            mask_prior = self.mask_adapter(torch.cat((mask, boundary), dim=1))
            pre_base = self.ffn1(norm1_1_x, x_spatial)
            pre_cond = self.ffn1(norm1_1_x, x_spatial + mask_prior)
            x = x + torch.tanh(self.pre_scale) * (pre_cond - pre_base)

        norm1_x = self.norm1(x)
        if self.training and self.use_gradient_checkpoint and norm1_x.requires_grad:
            attn_out = checkpoint(
                lambda inp: self.attn(inp, pos, mask_context),
                norm1_x,
                use_reentrant=False,
            )
        else:
            attn_out = self.attn(norm1_x, pos, mask_context)
        x = x + attn_out

        norm2_x = self.norm2(x)
        if self.training and self.use_gradient_checkpoint and norm2_x.requires_grad:
            ffn_out = checkpoint(
                lambda inp, spatial: self.ffn(inp, spatial),
                norm2_x,
                x_spatial,
                use_reentrant=False,
            )
        else:
            ffn_out = self.ffn(norm2_x, x_spatial)
        x = x + ffn_out

        return {0:x, 1:pos, 2:mask}
        




##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.gproj1 = nn.Conv2d(in_c, embed_dim, kernel_size=3,stride=1,padding=1,bias=bias)

    def forward(self, x):

        x = self.gproj1(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(
        self,
        n_feat,
        use_bias=False,
        use_mrda=False,
        residual_scale_init=0.0,
    ):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )
        self.proj = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=1, padding=1, groups=n_feat * 2, bias=False)
        self.mrda_branch = None
        self.residual_scale = None
        if use_mrda:
            self.mrda_branch = MRDADownsampleAdapter(
                in_channels=n_feat,
                out_channels=n_feat * 2,
                bias=use_bias,
            )
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x, mask):
        base = self.body(x)
        if self.mrda_branch is None:
            return base

        scale = torch.tanh(self.residual_scale)
        mrda_delta = self.mrda_branch(x, mask, base=base, return_delta=True)
        return base + scale * mrda_delta



class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x, mask):
        return self.body(x)


##########################################################################
##---------- PATINA Generator -----------------------
class SEM(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 config=None,
                 ):

        super(SEM, self).__init__()

        enable_stage3plus_precondition = bool(int(getattr(config, 'PATINA_PRECONDITION_ENABLE', 0) or 0))
        enable_stage3plus_route = bool(int(getattr(config, 'PATINA_MASK_ROUTE_ENABLE', 0) or 0))
        self.enable_skip_adapter = bool(int(getattr(config, 'PATINA_SKIP_FUSION_ENABLE', 0) or 0))
        self.enable_refinement = bool(int(getattr(config, 'PATINA_REFINEMENT_ENABLE', 0) or 0))
        precondition_scale_init = float(getattr(config, 'PATINA_PRECONDITION_SCALE_INIT', 0.0) or 0.0)
        route_scale_init = float(getattr(config, 'PATINA_MASK_ROUTE_SCALE_INIT', 0.0) or 0.0)
        skip_adapter_scale_init = float(getattr(config, 'PATINA_SKIP_ADAPTER_RES_SCALE_INIT', 0.0) or 0.0)
        refinement_scale_init = float(getattr(config, 'PATINA_REFINEMENT_RES_SCALE_INIT', 0.0) or 0.0)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=False,
                             enable_mask_route=False) for i in range(num_blocks[0])])

        mrda_bias = bool(int(getattr(config, 'MRDA_USE_BIAS', 0) or 0))
        use_mrda_stage1 = bool(int(getattr(config, 'MRDA_STAGE1_ENABLE', 1) or 0))
        use_mrda_stage2 = bool(int(getattr(config, 'MRDA_STAGE2_ENABLE', 1) or 0))
        use_mrda_stage3 = bool(int(getattr(config, 'MRDA_STAGE3_ENABLE', 1) or 0))

        self.down1_2 = Downsample(
            dim,
            use_bias=mrda_bias,
            use_mrda=use_mrda_stage1,
            residual_scale_init=float(getattr(config, 'MRDA_STAGE1_RES_SCALE_INIT', 0.04) or 0.04),
        )  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=False,
                             enable_mask_route=False) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(
            int(dim * 2 ** 1),
            use_bias=mrda_bias,
            use_mrda=use_mrda_stage2,
            residual_scale_init=float(getattr(config, 'MRDA_STAGE2_RES_SCALE_INIT', 0.08) or 0.08),
        )  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=enable_stage3plus_precondition,
                             enable_mask_route=enable_stage3plus_route,
                             precondition_scale_init=precondition_scale_init,
                             route_scale_init=route_scale_init) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(
            int(dim * 2 ** 2),
            use_bias=mrda_bias,
            use_mrda=use_mrda_stage3,
            residual_scale_init=float(getattr(config, 'MRDA_STAGE3_RES_SCALE_INIT', 0.12) or 0.12),
        )  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=enable_stage3plus_precondition,
                             enable_mask_route=enable_stage3plus_route,
                             precondition_scale_init=precondition_scale_init,
                             route_scale_init=route_scale_init) for i in range(num_blocks[3])])
        self.dfcc_latent = None
        if bool(int(getattr(config, 'DFCC_LATENT_ENABLE', 1) or 0)):
            self.dfcc_latent = DFCCResidualBlock(
                channels=int(dim * 2 ** 3),
                input_shape=int(getattr(config, 'DFCC_LATENT_INPUT_SHAPE', 32) or 32),
                groups=int(getattr(config, 'DFCC_GROUPS', 1) or 1),
                residual_scale_init=float(getattr(config, 'DFCC_LATENT_RES_SCALE_INIT', 0.08) or 0.08),
                fft_norm=str(getattr(config, 'DFCC_FFT_NORM', 'ortho') or 'ortho'),
                mask_context_kernel=int(getattr(config, 'DFCC_MASK_CONTEXT_KERNEL', 5) or 5),
            )
        self.lcbc_latent = None
        if bool(int(getattr(config, 'LCBC_LATENT_ENABLE', 1) or 0)):
            self.lcbc_latent = LCBCAdapter(
                channels=int(dim * 2 ** 3),
                embed_dim=int(getattr(config, 'LCBC_LATENT_EMBED_DIM', 96) or 96),
                softmax_scale=float(getattr(config, 'LCBC_SOFTMAX_SCALE', 10.0) or 10.0),
                residual_scale_init=float(getattr(config, 'LCBC_LATENT_RES_SCALE_INIT', 0.10) or 0.10),
            )
        self.latent_branch_mixer = None
        if (
            self.dfcc_latent is not None
            and self.lcbc_latent is not None
            and bool(int(getattr(config, 'LATENT_BRANCH_MIXER_ENABLE', 1) or 0))
        ):
            self.latent_branch_mixer = LatentBranchMixer(
                channels=int(dim * 2 ** 3),
                num_branches=2,
                temperature_init=float(getattr(config, 'LATENT_BRANCH_MIXER_TEMPERATURE', 1.0) or 1.0),
            )

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.skip_fusion_level3 = MaskGuidedSkipFusion(
            decoder_channels=int(dim * 2 ** 2),
            encoder_channels=int(dim * 2 ** 2),
            out_channels=int(dim * 2 ** 2),
        )
        self.skip_adapter_level3 = PretrainedSkipResidualAdapter(
            decoder_channels=int(dim * 2 ** 2),
            encoder_channels=int(dim * 2 ** 2),
            out_channels=int(dim * 2 ** 2),
            residual_scale_init=skip_adapter_scale_init,
        )
        self.dfcc_decoder_level3 = None
        if bool(int(getattr(config, 'DFCC_DECODER3_ENABLE', 1) or 0)):
            self.dfcc_decoder_level3 = DFCCResidualBlock(
                channels=int(dim * 2 ** 2),
                input_shape=int(getattr(config, 'DFCC_DECODER3_INPUT_SHAPE', 64) or 64),
                groups=int(getattr(config, 'DFCC_GROUPS', 1) or 1),
                residual_scale_init=float(getattr(config, 'DFCC_DECODER3_RES_SCALE_INIT', 0.05) or 0.05),
                fft_norm=str(getattr(config, 'DFCC_FFT_NORM', 'ortho') or 'ortho'),
                mask_context_kernel=int(getattr(config, 'DFCC_MASK_CONTEXT_KERNEL', 5) or 5),
            )
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=enable_stage3plus_precondition,
                             enable_mask_route=enable_stage3plus_route,
                             precondition_scale_init=precondition_scale_init,
                             route_scale_init=route_scale_init) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.skip_fusion_level2 = MaskGuidedSkipFusion(
            decoder_channels=int(dim * 2 ** 1),
            encoder_channels=int(dim * 2 ** 1),
            out_channels=int(dim * 2 ** 1),
        )
        self.skip_adapter_level2 = PretrainedSkipResidualAdapter(
            decoder_channels=int(dim * 2 ** 1),
            encoder_channels=int(dim * 2 ** 1),
            out_channels=int(dim * 2 ** 1),
            residual_scale_init=skip_adapter_scale_init,
        )
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=False,
                             enable_mask_route=False) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.skip_fusion_level1 = MaskGuidedSkipFusion(
            decoder_channels=dim,
            encoder_channels=dim,
            out_channels=int(dim * 2 ** 1),
        )
        self.skip_adapter_level1 = PretrainedSkipResidualAdapter(
            decoder_channels=dim,
            encoder_channels=dim,
            out_channels=int(dim * 2 ** 1),
            residual_scale_init=skip_adapter_scale_init,
        )

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=False,
                             enable_mask_route=False) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             enable_mask_precondition=False,
                             enable_mask_route=False) for i in range(num_refinement_blocks)])

        
        self.output_before = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.refinement_head = RefinementResidualHead(
            channels=int(dim * 2 ** 1),
            bottleneck_channels=dim,
            residual_scale_init=refinement_scale_init,
        )
        self.output = nn.Sequential(nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                                    )        


    def forward(self, inp_img, mask_whole, mask_half, mask_quarter,mask_tiny, pos1, pos2, pos3, pos4, pos1_dec):
        
        inp_enc_level1 = self.patch_embed(torch.cat((inp_img,mask_whole),dim=1))

        out_enc_level1 = self.encoder_level1({0:inp_enc_level1, 1:pos1, 2:mask_whole})

        inp_enc_level2 = self.down1_2(out_enc_level1[0],mask_whole)
        out_enc_level2 = self.encoder_level2({0:inp_enc_level2, 1:pos2, 2:mask_half})

        inp_enc_level3 = self.down2_3(out_enc_level2[0],mask_half)
        out_enc_level3 = self.encoder_level3({0:inp_enc_level3, 1:pos3, 2:mask_quarter})

        inp_enc_level4 = self.down3_4(out_enc_level3[0],mask_quarter)

        latent = self.latent({0:inp_enc_level4, 1:pos4, 2:mask_tiny})
        latent_feature = latent[0]
        latent_deltas = []
        if self.dfcc_latent is not None:
            latent_deltas.append(self.dfcc_latent(latent_feature, mask_tiny, return_delta=True))
        if self.lcbc_latent is not None:
            latent_deltas.append(self.lcbc_latent(latent_feature, mask_tiny, return_delta=True))
        if latent_deltas:
            if self.latent_branch_mixer is not None and len(latent_deltas) > 1:
                latent_feature = latent_feature + self.latent_branch_mixer(latent_feature, latent_deltas, mask_tiny)
            else:
                latent_mixed = latent_deltas[0]
                for delta in latent_deltas[1:]:
                    latent_mixed = latent_mixed + delta
                latent_feature = latent_feature + latent_mixed
        latent = {0: latent_feature, 1: pos4, 2: mask_tiny}

        inp_dec_level3 = self.up4_3(latent[0],mask_tiny)
        skip_level3_decoder = inp_dec_level3
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3[0]], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        if self.enable_skip_adapter:
            inp_dec_level3 = self.skip_adapter_level3(skip_level3_decoder, out_enc_level3[0], inp_dec_level3, mask_quarter)
        if self.dfcc_decoder_level3 is not None:
            inp_dec_level3 = self.dfcc_decoder_level3(inp_dec_level3, mask_quarter)
        out_dec_level3 = self.decoder_level3({0:inp_dec_level3, 1:pos3, 2:mask_quarter})

        inp_dec_level2 = self.up3_2(out_dec_level3[0],mask_quarter)
        skip_level2_decoder = inp_dec_level2
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2[0]], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        if self.enable_skip_adapter:
            inp_dec_level2 = self.skip_adapter_level2(skip_level2_decoder, out_enc_level2[0], inp_dec_level2, mask_half)
        out_dec_level2 = self.decoder_level2({0:inp_dec_level2, 1:pos2, 2:mask_half})

        inp_dec_level1 = self.up2_1(out_dec_level2[0],mask_half)
        skip_level1_decoder = inp_dec_level1
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1[0]], 1)
        if self.enable_skip_adapter:
            inp_dec_level1 = self.skip_adapter_level1(skip_level1_decoder, out_enc_level1[0], inp_dec_level1, mask_whole)
        out_dec_level1 = self.decoder_level1({0:inp_dec_level1, 1:pos1_dec, 2:mask_whole})

        final_feature = out_dec_level1[0]
        if self.enable_refinement:
            refined_feature = self.refinement({0:final_feature, 1:pos1_dec, 2:mask_whole})[0]
            refinement_bottleneck = self.output_before(refined_feature)
            final_feature = self.refinement_head(final_feature, refinement_bottleneck, mask_whole)
        out_dec_level1 = self.output(final_feature)

        out_dec_level1 = (torch.tanh(out_dec_level1) + 1) / 2
        return out_dec_level1
        
        
        
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, enable_mask_route=False, route_scale_init=0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.enable_mask_route = bool(enable_mask_route) and dim >= 192
        if self.enable_mask_route:
            self.mask_router = nn.Sequential(
                    nn.Conv2d(dim + 2, dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid(),
            )
            self.mask_projector = nn.Conv2d(2, dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.hole_pos_bias = nn.Parameter(torch.zeros(1, 1, dim))
            self.boundary_pos_bias = nn.Parameter(torch.zeros(1, 1, dim))
            self.route_scale = nn.Parameter(torch.tensor(float(route_scale_init)))
        else:
            self.mask_router = None
            self.mask_projector = None
            self.hole_pos_bias = None
            self.boundary_pos_bias = None
            self.route_scale = None
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
        self.mamba2 = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        
    
    @ _disabled_cuda_autocast()
    def forward(self, x, pe, mask):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        B, C, H, W = x.shape
        mask = resize_mask_like(mask, x)
        if self.enable_mask_route:
            mask_context = build_mask_context(mask, kernel_size=5)
            boundary = torch.clamp(mask_context - mask, 0.0, 1.0)
            route_tokens = torch.cat((mask_context, boundary), dim=1)
            route = self.mask_router(torch.cat((x, route_tokens), dim=1))
            x = x + torch.tanh(self.route_scale) * route * self.mask_projector(route_tokens)
        else:
            mask_context = mask
            boundary = mask.new_zeros(mask.shape)


        reversed_x1 = x.clone()

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        x2 = x.transpose(-1,-2)

  
        reversed_x2 = x2.clone()

        reversed_x1[:,:,1::2,:] = x[:,:,1::2,:].flip(-1)
        
        reversed_x2[:,:,1::2,:] = x2[:,:,1::2,:].flip(-1)

        #### add positional embedding
        x1_flat = reversed_x1.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x1_flat = x1_flat + pe[:n_tokens, :]

        x2_flat = reversed_x2.reshape(B, C, n_tokens).transpose(-1, -2)
        
        x2_flat = x2_flat + pe[:n_tokens, :]
        if self.enable_mask_route:
            mask_x1 = mask_context.reshape(B, 1, n_tokens).transpose(-1, -2)
            boundary_x1 = boundary.reshape(B, 1, n_tokens).transpose(-1, -2)
            mask_x2 = mask_context.transpose(-1, -2).reshape(B, 1, n_tokens).transpose(-1, -2)
            boundary_x2 = boundary.transpose(-1, -2).reshape(B, 1, n_tokens).transpose(-1, -2)
            x1_flat = x1_flat + mask_x1 * self.hole_pos_bias + boundary_x1 * self.boundary_pos_bias
            x2_flat = x2_flat + mask_x2 * self.hole_pos_bias + boundary_x2 * self.boundary_pos_bias
        ###### end adding positional embedding

        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm)
        
        x2_norm = self.norm2(x2_flat)
        x2_mamba = self.mamba2(x2_norm)

        out1 = x1_mamba.transpose(-1, -2).reshape(B, C, H, W)
        out1_clone2 = out1
        out1_clone2[:,:,1::2,:] = out1[:,:,1::2,:].flip(-1)
        
        out2 = x2_mamba.transpose(-1, -2).reshape(B, C, W, H)
        out2_clone2 = out2
        out2_clone2[:,:,1::2,:] = out2[:,:,1::2,:].flip(-1)
        out2_clone2 = out2_clone2.transpose(-1,-2)
        
        out = out1_clone2 + out2_clone2
        

        return out
        
