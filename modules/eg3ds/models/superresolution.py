# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.eg3ds.models.networks_stylegan2 import Conv2dLayer, SynthesisLayer, ToRGBLayer
from modules.eg3ds.torch_utils.ops import upfirdn2d
from modules.eg3ds.torch_utils import misc

from modules.eg3ds.models.networks_stylegan2 import SynthesisBlock
from modules.eg3ds.models.networks_stylegan3 import SynthesisLayer as AFSynthesisLayer
from utils.commons.hparams import hparams


#----------------------------------------------------------------------------

# for 512x512 generation
class SuperresolutionHybrid8X(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlock(channels, 128, w_dim=512, resolution=256,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=512,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

#----------------------------------------------------------------------------

# for 256x256 generation

class SuperresolutionHybrid4X(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0
        self.sr_antialias = sr_antialias
        self.input_resolution = 128
        self.block0 = SynthesisBlockNoUp(channels, 128, w_dim=512, resolution=128,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=256,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] < self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

#----------------------------------------------------------------------------

# for 128 x 128 generation

class SuperresolutionHybrid2X(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 128

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 64
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlockNoUp(channels, 128, w_dim=512, resolution=64,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=128,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

#----------------------------------------------------------------------------

# TODO: Delete (here for backwards compatibility with old 256x256 models)

class SuperresolutionHybridDeepfp32(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0

        self.input_resolution = 128
        self.block0 = SynthesisBlockNoUp(channels, 128, w_dim=512, resolution=128,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=256,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] < self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

#----------------------------------------------------------------------------


class SynthesisBlockNoUp(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution,
                conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        # if img is not None:
            # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            # img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


#----------------------------------------------------------------------------
# for 512x512 generation
class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.act = nn.ReLU(inplace=False)
        # self.act = nn.LeakyReLU(inplace=False) # run3
        # self.norm1 = nn.BatchNorm2d(in_features, affine=True)
        # self.norm2 = nn.BatchNorm2d(in_features, affine=True)

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        out = out + x
        return out

    # def forward(self, x):
    #     out = self.norm1(x)
    #     out = F.relu(out)
    #     out = self.conv1(out)
    #     out = self.norm2(out)
    #     out = F.relu(out)
    #     out = self.conv2(out)
    #     out = x + out
    #     return out


class LargeSynthesisBlock0(nn.Module):
    def __init__(self, channels, use_fp16, **block_kwargs):
        super().__init__()
        self.block = SynthesisBlock(channels, 256, w_dim=512, resolution=256,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.resblocks = nn.Sequential(*[
            ResBlock2d(256, kernel_size=3, padding=1) for _ in range(hparams['resblocks_in_large_sr'])
        ])
        self.to_rgb = nn.Conv2d(256, 3, kernel_size=1)

    def forward(self, x, rgb, ws, **block_kwargs):
        x, rgb = self.block(x, rgb, ws, **block_kwargs)
        x = self.resblocks(x)
        rgb = rgb + self.to_rgb(x)
        return x, rgb

class LargeSynthesisBlock1(nn.Module):
    def __init__(self, use_fp16, **block_kwargs):
        super().__init__()
        self.block = SynthesisBlock(256, 128, w_dim=512, resolution=512,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.resblocks = nn.Sequential(*[
            ResBlock2d(128, kernel_size=3, padding=1) for _ in range(hparams['resblocks_in_large_sr'])
        ])
        self.to_rgb = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x, rgb, ws, **block_kwargs):
        x, rgb = self.block(x, rgb, ws, **block_kwargs)
        x = self.resblocks(x)
        rgb = rgb + self.to_rgb(x)
        return x, rgb
    
class AFBlockAdapter(nn.Module):
    """
    An adapter to make a StyleGAN3 AFSynthesisLayer behave like a StyleGAN2 SynthesisBlock.
    It takes (x, rgb, ws) and returns (x, rgb_out), hiding the internal complexity of
    calling the main alias-free layer and its corresponding to-rgb layer.
    """
    def __init__(self, af_layer, torgb_layer, resample_filter):
        super().__init__()
        self.af_layer = af_layer
        self.torgb_layer = torgb_layer
        self.register_buffer('resample_filter', resample_filter)

    def forward(self, x, rgb, ws, **block_kwargs):
        # AFSynthesisLayer expects a single w, not the whole ws stack.
        # We'll use the first w in the sequence for this block.
        # The original StyleGAN2 block iterates through ws, but here we simplify
        # as the calling context only provides 3 ws vectors anyway.
        w = ws[:, 0, :]

        # Filter forward kwargs to be compatible with AFSynthesisLayer's forward method.
        af_forward_kwargs = {}
        if 'force_fp32' in block_kwargs:
            af_forward_kwargs['force_fp32'] = block_kwargs['force_fp32']
        if 'update_emas' in block_kwargs:
            af_forward_kwargs['update_emas'] = block_kwargs['update_emas']

        # Main feature processing
        x = self.af_layer(x, w, **af_forward_kwargs)
        
        # Get the RGB output for this stage
        new_rgb = self.torgb_layer(x, w, **af_forward_kwargs)

        # Combine with previous RGB output, upsampling if necessary
        if rgb is not None:
            if rgb.shape[-1] != new_rgb.shape[-1]:
                rgb = upfirdn2d.upsample2d(rgb, self.resample_filter)
            rgb = rgb + new_rgb
        else:
            rgb = new_rgb
            
        return x, rgb

class SuperresolutionHybrid8XDC(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, large_sr=False, lora_args=None, **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        
        self.lora_args = lora_args
        self.sr_arch = hparams.get('sr_arch', 'stylegan2')
        
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

        if self.sr_arch == 'stylegan2':
            if large_sr is True:
                self.block0 = LargeSynthesisBlock0(channels, use_fp16=sr_num_fp16_res > 0, **block_kwargs)
                self.block1 = LargeSynthesisBlock1(use_fp16=sr_num_fp16_res > 0, **block_kwargs)
            else:
                self.block0 = SynthesisBlock(channels, 256, w_dim=512, resolution=256,
                        img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), lora_args=self.lora_args, **block_kwargs)
                self.block1 = SynthesisBlock(256, 128, w_dim=512, resolution=512,
                        img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), lora_args=self.lora_args, **block_kwargs)
        elif self.sr_arch == 'stylegan3':
            w_dim = 512
            # Filter kwargs to be compatible with AFSynthesisLayer by creating an allowlist
            # of arguments it can accept. This is safer than blacklisting.
            valid_af_kwargs = [
                'conv_kernel', 'filter_size', 'lrelu_upsampling', 'use_radial_filters',
                'conv_clamp', 'magnitude_ema_beta'
            ]
            af_kwargs = {k: v for k, v in block_kwargs.items() if k in valid_af_kwargs}
            # Force lrelu_upsampling to 1, so upsampling is controlled by sampling_rate parameters
            af_kwargs['lrelu_upsampling'] = 1

            # Define the raw alias-free layers
            # Stage 1: 128 -> 256
            block0_af = AFSynthesisLayer(w_dim, is_torgb=False, is_critically_sampled=False, use_fp16=use_fp16,
                in_channels=channels, out_channels=256, in_size=128, out_size=256,
                in_sampling_rate=128, out_sampling_rate=256, in_cutoff=64, out_cutoff=128,
                in_half_width=64, out_half_width=128, **af_kwargs)
            torgb0 = AFSynthesisLayer(w_dim, is_torgb=True, is_critically_sampled=False, use_fp16=use_fp16,
                in_channels=256, out_channels=3, in_size=256, out_size=256,
                in_sampling_rate=256, out_sampling_rate=256, in_cutoff=128, out_cutoff=128,
                in_half_width=128, out_half_width=128, **af_kwargs)
            
            # Stage 2: 256 -> 512
            block1_af = AFSynthesisLayer(w_dim, is_torgb=False, is_critically_sampled=False, use_fp16=use_fp16,
                in_channels=256, out_channels=128, in_size=256, out_size=512,
                in_sampling_rate=256, out_sampling_rate=512, in_cutoff=128, out_cutoff=256,
                in_half_width=128, out_half_width=256, **af_kwargs)
            torgb1 = AFSynthesisLayer(w_dim, is_torgb=True, is_critically_sampled=True, use_fp16=use_fp16,
                in_channels=128, out_channels=3, in_size=512, out_size=512,
                in_sampling_rate=512, out_sampling_rate=512, in_cutoff=256, out_cutoff=256,
                in_half_width=256, out_half_width=256, **af_kwargs)

            # Wrap the AF layers in the adapter to provide a StyleGAN2-like interface
            self.block0 = AFBlockAdapter(block0_af, torgb0, self.resample_filter)
            self.block1 = AFBlockAdapter(block1_af, torgb1, self.resample_filter)
        else:
            raise NotImplementedError(f"Unknown SR architecture: {self.sr_arch}")


    def forward(self, rgb, x, ws, **block_kwargs):
        # This forward pass now works for both StyleGAN2 and the adapted StyleGAN3 blocks
        # as they share the same interface.
        if self.sr_arch == 'stylegan3':
            # StyleGAN3 adapter blocks expect 2 ws vectors each, but are called sequentially.
            # The total ws vectors needed is 2.
            ws = ws[:, -1:, :].repeat(1, 2, 1)
        else: # stylegan2
            ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb
#----------------------------------------------------------------------------