import torch
import torch.nn as nn
from torch.nn.functional import grid_sample, pad
from torchvision.models.optical_flow import raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
from einops import repeat


class InputPadder:
    # originally from the GMA repo.
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='kitti'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    

class PreprocessAndGetOutput(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def preprocess(self, x):
        return x * 2 - 1
    
    def forward(self, img0, img1, final_only=True):
        img0, img1 = self.preprocess(img0), self.preprocess(img1)
        padder = InputPadder(img0.shape)
        img0, img1 = padder.pad(img0, img1)
        flow_pred = self.model(img0, img1)
        if final_only:
            flow_pred = flow_pred[-1]
            flow_pred = padder.unpad(flow_pred)
        else:
            for i, pred_i in enumerate(flow_pred):
                flow_pred[i] = padder.unpad(pred_i)
        return flow_pred


def getFlowModel(model_type='RAFT_Large', model_path=None):
    model_type = model_type.lower()
    if 'raft' in model_type:
        if 'small' in model_type:
            model = raft_small(Raft_Small_Weights.DEFAULT, progress=False)
        else:  # if 'large' in model_type:
            model = raft_large(Raft_Large_Weights.DEFAULT, progress=False)
        model = PreprocessAndGetOutput(model)
    else:  # GMA, Flow1d, GMFlow ,FlowFormer++, etc.
        raise NotImplementedError(f'import of Flow Model {model_type} not implemented.')
    return model


class BackWarp(nn.Module):
    def __init__(self, clip=False, interpolation='bilinear', align_corners=False):
        super().__init__()
        self.clip = clip
        self.interpolation = interpolation
        self.align_corners = align_corners

    def forward(self, img, flow):
        b, _, h, w = img.shape
        gridY, gridX = torch.meshgrid(torch.arange(h), torch.arange(w))
        gridX, gridY = gridX.to(img.device), gridY.to(img.device)

        u = flow[:, 0]  # W
        v = flow[:, 1]  # H

        x = repeat(gridX, 'h w -> b h w', b=b).float() + u
        y = repeat(gridY, 'h w -> b h w', b=b).float() + v

        # normalize
        x = (x / w) * 2 - 1
        y = (y / h) * 2 - 1

        # stacking X and Y
        grid = torch.stack((x, y), dim=-1)

        if self.clip:  # clip flow values exceeding range to max.
            output = grid_sample(img, grid, mode=self.interpolation, align_corners=self.align_corners, padding_mode='border')
        else:
            output = grid_sample(img, grid, mode=self.interpolation, align_corners=self.align_corners, padding_mode='zeros')
        return output

