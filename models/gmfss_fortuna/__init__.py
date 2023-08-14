import pathlib
from utils import load_file_from_github_release
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .GMFSS_Fortuna_arch import Model as GMFSS
from .GMFSS_Fortuna_union_arch import Model as GMFSS_Union


GLOBAL_MODEL_TYPE = pathlib.Path(__file__).parent.name
MODELS_PATH_CONFIG = {
    "GMFSS_fortuna_union": {
        "ifnet": ("rife", "rife46.pth"),
        "flownet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_flownet.pkl"),
        "metricnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_metric.pkl"),
        "feat_ext": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_feat.pkl"),
        "fusionnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_fusionnet.pkl")
    },
    "GMFSS_fortuna": {
        "flownet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_flownet.pkl"),
        "metricnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_metric.pkl"),
        "feat_ext": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_feat.pkl"),
        "fusionnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_fusionnet.pkl")
    }
}

class CommonModelInference(nn.Module):
    def __init__(self, model_type):
        super(CommonModelInference, self).__init__()
        self.model = GMFSS_Union() if "union" in model_type else GMFSS()
        self.model.eval()
        self.model.device()
        _model_path_config = MODELS_PATH_CONFIG[model_type]
        self.model.load_model({
            key: load_file_from_github_release(*_model_path_config[key])
            for key in _model_path_config
        })

    def forward(self, I0, I1, timestep, scale=1.0):
        n, c, h, w = I0.shape
        tmp = max(64, int(64 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        (
            flow01,
            flow10,
            metric0,
            metric1,
            feat11,
            feat12,
            feat13,
            feat21,
            feat22,
            feat23,
        ) = self.model.reuse(I0, I1, scale)

        output = self.model.inference(
            I0,
            I1,
            flow01,
            flow10,
            metric0,
            metric1,
            feat11,
            feat12,
            feat13,
            feat21,
            feat22,
            feat23,
            timestep,
        )
        return output[:, :, :h, :w]

class GMFSS_Fortuna_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (MODELS_PATH_CONFIG.keys(), ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGES", )
    FUNCTION = "vfi"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        batch_size: typing.SupportsInt = 1,
        multipler: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1,
        optional_interpolation_states: typing.Optional[typing.List[bool]] = None
    ):
        global model
        model = CommonModelInference(model_type=ckpt_name)
        model.eval().cuda()
        frames.cuda()
        
        frame_dict = {
            str(i): frames[i].unsqueeze(0) for i in range(frames.shape[0])
        }

        if optional_interpolation_states is None:
            interpolation_states = [True] * (frames.shape[0] - 1)
        else:
            interpolation_states = optional_interpolation_states

        enabled_former_idxs = [i for i, state in enumerate(interpolation_states) if state]
        former_idxs_loader = DataLoader(enabled_former_idxs, batch_size=batch_size)
        
        for former_idxs_batch in former_idxs_loader:
            for middle_i in range(1, multipler):
                _middle_frames = model(
                    frames[former_idxs_batch], 
                    frames[former_idxs_batch + 1], 
                    timestep=middle_i/multipler,
                    scale=scale_factor
                )
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
