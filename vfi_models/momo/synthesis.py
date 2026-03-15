import torch
import torch.nn as nn
from .flow import BackWarp
from einops import rearrange
import numpy as np
from torch.nn.functional import interpolate, l1_loss


class SynthesisNet(nn.Module):
    def __init__(
        self,
        latent_dim=32,
        recurrent_min_res=64,
        normalize_inputs=True,
        align_corners=False,
        padding='replicate',
        interpolation='bicubic',
        act=nn.GELU,
        antialias=True,
        multi_scale_loss=True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.recurrent_min_res = recurrent_min_res
        self.normalize_inputs = normalize_inputs
        self.align_corners = align_corners
        self.padding = padding
        self.interpolation = interpolation
        self.antialias = antialias
        self.multi_scale_loss = multi_scale_loss
        
        self.bwarp = BackWarp(interpolation=interpolation, align_corners=align_corners)

        dim = latent_dim * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, latent_dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1, padding_mode=padding),
        )

        self.decoder = nn.Sequential(
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=padding),
            act(),
            nn.Conv2d(dim, 4, 3, 1, 1, padding_mode=padding),
        )

        self.blender = UNet(in_channels=4 + 3 + latent_dim * 2, out_channels=dim, n_lvls=2, dim=dim, act=act)

    def preprocess(self, x, eps=1e-8, stats=None):
        if self.normalize_inputs:
            if stats is None:
                x_flat = x.view(x.shape[0], -1)
                _mean, _std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
                while len(_mean.shape) < len(x.shape):
                    _mean, _std = _mean.unsqueeze(-1), _std.unsqueeze(-1)
                normalized_x = (x - _mean) / _std
                return normalized_x, (_mean, _std)
            else:
                _mean, _std = stats
                normalized_x = (x - _mean) / _std
                return normalized_x, None
        return x * 2 - 1, None
    
    def postprocess(self, x, stats=None):
        if self.normalize_inputs and stats is not None:
            _mean, _std = stats
            return torch.clamp((x * _std) + _mean, 0, 1)
        return torch.clamp((x + 1) / 2, 0, 1)

    def get_n_lvls(self, size):
        lvls = int(np.ceil(np.log2(min(size) / self.recurrent_min_res)))
        return lvls + 1

    def decode2rgb(self, xt, warped_xt_rgb):
        # predict the blending parameters
        output = self.decoder(xt)
        
        # synthesis from output
        res_rgb, blend_w = output.split([3, 1], dim=1)
        blend_w = torch.sigmoid(blend_w)
        blend_w = torch.stack([blend_w, 1 - blend_w], dim=2)
        synth_out = torch.sum(warped_xt_rgb * blend_w, dim=2) + res_rgb

        return synth_out
    
    def forward(self, x, flows, target=None, loss_fn=l1_loss, **kwargs):
        x, x_norm_stats = self.preprocess(rearrange(x, 'b c f h w -> b (f c) h w'))
        x = rearrange(x, 'b (f c) h w -> (f b) c h w', f=2)
        flows = rearrange(flows, 'b (f c) h w -> (f b) c h w', f=2)
        n_lvls = self.get_n_lvls(flows.shape[-2:])  # get number of levels

        for i in range(n_lvls - 1, -1, -1):
            # resize to current level.
            scale_factor = 1 / (2 ** i)
            x_lvl = interpolate(x, scale_factor=scale_factor, mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias)
            flows_lvl = interpolate(flows, scale_factor=scale_factor, mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias) * scale_factor
            
            # warp RGB image
            warped_xt0_rgb, warped_xt1_rgb = self.bwarp(x_lvl, flows_lvl).chunk(2, dim=0)
            warped_xt_rgb = torch.stack([warped_xt0_rgb, warped_xt1_rgb], dim=2)

            # extract features
            enc_x_lvl = self.encoder(x_lvl)
            if i == n_lvls - 1:
                xt = (warped_xt0_rgb + warped_xt1_rgb) / 2
            else:
                xt = interpolate(xt, size=flows_lvl.shape[-2:], mode=self.interpolation, align_corners=self.align_corners, antialias=self.antialias)
            
            # warping
            warped_xs_lvl = self.bwarp(enc_x_lvl, flows_lvl)
            warped_xs_lvl = rearrange(warped_xs_lvl, '(f b) c h w -> b (f c) h w', f=2)
            
            # predict the rgb frame
            xt = self.blender(torch.cat([xt, warped_xs_lvl, rearrange(flows_lvl, '(f b) c h w -> b (f c) h w', f=2)], dim=1))
            xt = self.decode2rgb(xt, warped_xt_rgb)

        # if training mode: return loss
        if self.training and target is not None and loss_fn is not None:
            loss = 0
            target = target * 2 - 1  # make it [-1, 1] range for LPIPS
            if self.normalize_inputs:
                xt = self.postprocess(xt, stats=x_norm_stats) * 2 - 1  # make it back to [-1, 1] range for LPIPS
            loss = loss_fn(xt, target)
            return loss

        return self.postprocess(xt, stats=x_norm_stats)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU, padding_mode='replicate'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, padding_mode=padding_mode),
            act(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding='same', padding_mode=padding_mode),
            act(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=nn.ReLU, padding_mode='replicate', interpolation='bicubic'):
        super().__init__()
        self.interpolation = interpolation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding='same', padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(2 * out_channels, out_channels, kernel_size, 1, padding='same', padding_mode=padding_mode)
        self.act = act()

    def forward(self, x, skip):
        _, _, h, w = skip.shape
        x = interpolate(x, size=(h, w), mode=self.interpolation)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(torch.cat((x, skip), 1)))
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_lvls=4,
        dim=32,
        max_dim=None,
        act=nn.ReLU,
        padding_mode='replicate',
        interpolation='bicubic',
    ) -> None:
        super().__init__()
        # levels: number of down / up blocks
        self.n_lvls = n_lvls

        # initial features
        self.in_feats = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, 1, 1, padding_mode=padding_mode),
            act(),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=padding_mode),
            act()
        )
        
        # downsample blocks
        down_blocks = []
        dim_tracker = [dim]
        for i in range(n_lvls):
            prev_dim = dim_tracker[i]
            next_dim = prev_dim * 2
            if max_dim is not None:
                bound_dim = min(max_dim, next_dim)
                if bound_dim != next_dim:
                    next_dim = bound_dim
            dim_tracker.append(next_dim)
            down_blocks.append(DownBlock(prev_dim, next_dim, kernel_size=3, act=act, padding_mode=padding_mode))
        self.down_blocks = nn.ModuleList(down_blocks)

        # upsample blocks
        up_blocks = []
        dim_tracker.reverse()
        for i in range(n_lvls):
            prev_dim = dim_tracker[i]
            next_dim = dim_tracker[i + 1]
            up_blocks.append(UpBlock(prev_dim, next_dim, kernel_size=3, act=act, padding_mode=padding_mode, interpolation=interpolation))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.to_out = nn.Conv2d(next_dim, out_channels, 3, 1, 1, padding_mode=padding_mode)

    def forward(self, x):
        mid_results = [self.in_feats(x)]
        for down_block in self.down_blocks:
            mid_results.append(down_block(mid_results[-1]))
        h = mid_results.pop()
        for up_block in self.up_blocks:
            h = up_block(h, mid_results.pop())
        out = self.to_out(h)
        return out
