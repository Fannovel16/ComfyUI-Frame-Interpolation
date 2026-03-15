# modified from the diffusers library.

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import get_down_block
from diffusers import UNet2DModel
from diffusers.models.resnet import Upsample2D, ResnetBlock2D
import math


@dataclass
class UNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: torch.FloatTensor


class ConvexUpUNet2DModel(ModelMixin, ConfigMixin):
    r"""
    UNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        add_attention (`bool', defaults to `False'): Whether to use the self-attention layer in the U-Net.
        class_embed_type (`str`, *optional*, defaults to None): The type of class embedding to use which is ultimately
            summed with the time embeddings. Choose from `None`, `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 4,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types: Tuple[str] = ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (256, 256, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        # downsampling images
        self.down_patch = nn.Sequential(
            nn.Conv2d(in_channels, int(block_out_channels[0] / 2), 8, 8),
            nn.SiLU(),
        )
        # downsampling flows
        self.down_latent = nn.Sequential(
            nn.Conv2d(out_channels, block_out_channels[0], 8, 8),
            nn.SiLU(),
        )
        # projection of downsampled images & flows to one representation
        self.proj_inputs = nn.Conv2d(block_out_channels[0] * 2, block_out_channels[0], kernel_size=1)
            
        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16, log=False)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # first residual block
        self.first_block = get_down_block(
            down_block_types[0],
            num_layers=layers_per_block,
            in_channels=block_out_channels[0],
            out_channels=block_out_channels[0],
            temb_channels=time_embed_dim,
            add_downsample=False,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            attention_head_dim=attention_head_dim,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

        # main middle model which outputs the coarsest flow map
        self.mid_model = UNet2DModel(
            sample_size=None,
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            down_block_types=down_block_types[1:],
            up_block_types=up_block_types[:-1],
            block_out_channels=block_out_channels[1:],
            add_attention=add_attention,
            class_embed_type=None,
        )

        # for convex upsampling
        # slightly customized UpBlock2D
        mask_w = 8 * 8 * 9 * 2
        self.out_up = UpMaskBlock2D(
            num_layers=layers_per_block + 1,
            in_channels=block_out_channels[0],
            out_channels=mask_w,
            prev_output_channel=out_channels,
            temb_channels=time_embed_dim,
            add_upsample=False,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    def convex_upsampling(self, flow, mask):
        # RAFT-style convex-upsampling
        b, _, h, w = flow.shape
        mask = mask.view(b, 2, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=3)
        up_flow = torch.nn.functional.unfold(flow, kernel_size=3, padding=1).view(b, 2, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=3)
        up_flow = up_flow.permute(0, 1, 2, 5, 3, 6, 4).reshape(b, self.config.out_channels, h * 8, w * 8)
        up_flow = up_flow * 8
        return up_flow

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        latents, x0, x1 = sample.split([self.config.out_channels, self.config.in_channels, self.config.in_channels], dim=1)
        
        # 3. x8 downsampling
        dx0, dx1 = self.down_patch(torch.cat([x0, x1], dim=0)).chunk(2, dim=0)
        dl = self.down_latent(latents)
        sample = torch.cat([dx0, dx1, dl], dim=1)
        sample = self.proj_inputs(sample)
        down_block_res_samples = (sample,)
        
        # 4. first residual block
        sample, res_samples = self.first_block(hidden_states=sample, temb=emb)
        down_block_res_samples += res_samples

        # 5. mid
        sample = self.mid_model(sample, timesteps).sample
        
        # 6. x8 convex-upsampling
        res_samples = down_block_res_samples[-len(self.out_up.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(self.out_up.resnets)]
        up_mask = self.out_up(sample, res_samples, emb)
        sample = self.convex_upsampling(sample, up_mask)
        
        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)


class UpMaskBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,  # mask weights
        out_channels: int,  # bidirectional flow: defaults to 4.
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample: bool = False,
        groups=32,
        eps=1e-6,
    ):
        super().__init__()
        resnets = []

        self.proj_in = nn.Identity()
        hidden_dim = int(math.ceil((prev_output_channel + in_channels) / resnet_groups) * resnet_groups)
        if hidden_dim != prev_output_channel + in_channels:
            self.proj_in = nn.Conv2d(prev_output_channel + in_channels, hidden_dim, 3, 1, 1)
        for i in range(num_layers):
            resnet_in_channels = hidden_dim if i == 0 else in_channels * 2
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(in_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
            self.proj_out = nn.Sequential(
                nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for i, resnet in enumerate(self.resnets):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if i == 0:
                hidden_states = self.proj_in(hidden_states)  # match number of dimensions

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        else:
            hidden_states = self.proj_out(hidden_states)

        return hidden_states