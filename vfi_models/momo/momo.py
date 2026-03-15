import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import l1_loss, interpolate
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np
from .unet import ConvexUpUNet2DModel
from torch.nn.functional import pad


class MoMo(nn.Module):
    def __init__(
        self,
        synth_model=None,
        dims=(256, 256, 512),
        T=1000,
        flow_scaler=128,
        prediction_type='sample',
        align_corners=False,
        clip_sample=True,
        max_rel_offset=1,
        beta_schedule='linear',
        use_attn=False,
        norm_in=True,
        padding='replicate',
        interpolation='bicubic',
        train_res=256,
    ) -> None:
        super().__init__()
        # synthesis model        
        self.synth_model = synth_model
        if self.synth_model is not None:
            for params in self.synth_model.parameters():
                params.requires_grad = False

        # U-Net
        self.dims = dims
        down_blocks = ["DownBlock2D"] * len(dims)
        up_blocks = ["UpBlock2D"] * len(dims)
        self.model = ConvexUpUNet2DModel(
            sample_size=None,
            in_channels=3,
            out_channels=4,
            down_block_types=tuple(down_blocks),
            up_block_types=tuple(up_blocks),
            block_out_channels=dims,
            add_attention=use_attn,
        )
        
        # ddpm scheduler
        assert prediction_type in ['epsilon', 'v_prediction', 'sample']
        self.prediction_type = prediction_type
        self.scheduler = DDPMScheduler(
            num_train_timesteps=T,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=max_rel_offset,
            timestep_spacing='trailing',
        )

        # other cfgs.
        self.flow_scaler = flow_scaler
        self.align_corners = align_corners
        self.norm_in = norm_in
        self.padding = padding
        self.interpolation = interpolation
        self.train_res = train_res

        # number of x2 up/down samples it goes through.
        # to avoid errors from spatial resolution mismatch in U-Net, compute number of downsamplings to consider.
        self.min_ds = 2 + len(dims)

    def prepare_latents(self, shape, **kwargs):
        b, c, h, w = shape
        noise = torch.randn(b, c, h, w, **kwargs)
        noise = noise * self.scheduler.init_noise_sigma
        return noise
    
    def preprocess(self, x, eps=1e-8):
        if self.norm_in:
            b = x.shape[0]
            x_flat = x.view(b, -1)
            _mean, _std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
            while len(_mean.shape) < len(x.shape):
                _mean, _std = _mean.unsqueeze(-1), _std.unsqueeze(-1)
            normalized_x = (x - _mean) / _std
            return normalized_x, (_mean, _std)
        else:
            return x * 2 - 1, None
    
    def postprocess(self, x, stats=None):
        if self.norm_in:
            _mean, _std = stats
            return (x * _std) + _mean
        else:
            return torch.clamp((x + 1) / 2, 0, 1)
    
    def normalize_flows(self, x):
        # flow values to a normalized range.
        assert self.training, 'normalization of flows would be used only during training.'
        x = x / self.flow_scaler
        return x

    def denormalize_flows(self, latent):
        # normalized flows to actual flow values
        latent = latent * self.flow_scaler
        return latent
    
    def ensure_resolution_fit(self, x, resize_to_fit=False, pad_to_fit_unet=False):
        # how to handle frames with resolutions that differ training data
        # resize to fit or naive forwarding w/o any fix.
        h, w = x.shape[-2:]
        if resize_to_fit:
            new_h, new_w = self.train_res, self.train_res
            if h > w:
                new_h = int(new_w / w * h)
            else:
                new_w = int(new_h / h * w)
            x = interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=self.align_corners, antialias=True)

        # ensure resolution to avoid errors that could arise from up/down-sampling in U-Net
        ds = 2 ** self.min_ds
        pad_size = None
        if pad_to_fit_unet:
            pad_h = int(np.ceil(x.shape[-2] / ds) * ds) - x.shape[-2]
            pad_w = int(np.ceil(x.shape[-1] / ds) * ds) - x.shape[-1]
            if pad_h == 0 and pad_w == 0:
                pad_to_fit_unet = False
            else:
                pad_size = [pad_w // 2, pad_w - pad_w // 2, 0, pad_h]
                x = pad(x, pad_size, mode=self.padding)
        else:
            new_h = int(np.round(x.shape[-2] / ds) * ds)
            new_w = int(np.round(x.shape[-1] / ds) * ds)
            x = interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=self.align_corners, antialias=True)
        
        return x, pad_size
    
    def restore_orig_resolution(self, x, orig_hw, pad_to_fit_unet=False, pad_size=None):
        if pad_to_fit_unet:
            assert pad_size is not None, 'pad size is not given! check the configuration.'
            cur_h, cur_w = x.shape[-2:]
            x = x[..., pad_size[2]: cur_h - pad_size[3], pad_size[0]: cur_w - pad_size[1]]
        out_h, out_w = x.shape[-2:]
        orig_h, orig_w = orig_hw
        scale_factor = torch.tensor([orig_w / out_w, orig_h / out_h], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1)
        scale_factor = torch.cat([scale_factor, scale_factor], dim=1)
        x = interpolate(x, size=(orig_h, orig_w), mode=self.interpolation, align_corners=self.align_corners) * scale_factor
        return x
    
    def forward(
        self,
        x,
        target=None,
        num_inference_steps=8,
        resize_to_fit=False,
        pad_to_fit_unet=False,
        loss_fn=l1_loss,
        **kwargs,
    ):
        orig_x = x
        x = rearrange(x, 'b c f h w -> b (f c) h w')
        x, x_norm_stats = self.preprocess(x)
        b, _, h, w = x.shape
        orig_hw = h, w

        # train mode
        if self.training:
            # scale flow values to match the scale of gaussian noise
            target = self.normalize_flows(target)

            # randomly insert noise
            noise = torch.randn_like(target)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,), device=noise.device).long()
            noisy_latent = self.scheduler.add_noise(target, noise, timesteps)

            # forwarding through model
            _input = torch.cat([noisy_latent, x], dim=1)
            pred = self.model(_input, timesteps).sample

            # target setting
            if self.prediction_type == 'epsilon':
                target = noise
            elif self.prediction_type == 'v_prediction':
                target = self.scheduler.get_velocity(target, noise, timesteps)
            
            # compute loss
            loss = loss_fn(pred, target)
            return loss
        
        # eval mode
        else:
            # ensure resolution fit and avoid errors
            x, pad_size = self.ensure_resolution_fit(x, resize_to_fit=resize_to_fit, pad_to_fit_unet=pad_to_fit_unet)

            # main generation process from here
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=x.device)
            timesteps = self.scheduler.timesteps
            
            # initial noise to start from.
            latents = self.prepare_latents(shape=(b, 4, x.shape[-2], x.shape[-1]), dtype=x.dtype, device=x.device)

            # iterative denoising
            for _, t in enumerate(timesteps):
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                _input = torch.cat([latent_model_input, x], dim=1)

                # prediction with diffusion model
                pred = self.model(_input, t).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents).prev_sample

            # rollback flow values to original scale.
            flows = self.denormalize_flows(latents)

            # postprocess and rollback resolution fitting
            flows = self.restore_orig_resolution(flows, orig_hw=orig_hw, pad_to_fit_unet=pad_to_fit_unet, pad_size=pad_size)
            
            # synthesize the final frame with flows
            xt = self.synth_model(orig_x, flows)
            return xt, flows
