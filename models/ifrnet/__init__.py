import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release
import typing
from .IFRNet_S_arch import IRFNet_S
from .IFRNet_L_arch import IRFNet_L

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["IFRNet_S_Vimeo90K.pth", "IFRNet_L_Vimeo90K.pth"]

class IFRNet_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "multipler": ("INT", {"default": 2, "min": 1, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
                "fast_mode": (["enabled", "disabled"], ),
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
        multipler: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        optional_interpolation_states: typing.Optional[list[bool]] = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        global model
        model = IRFNet_S() if 'S' in ckpt_name else IRFNet_L()
        model.load_state_dict(torch.load(model_path))
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
        
        for former_idx in enabled_former_idxs:
            for middle_i in range(1, multipler):
                _middle_frame = model(
                    frames[former_idx].unsqueeze(0), 
                    frames[former_idx + 1].unsqueeze(0), 
                    timestep=middle_i/multipler,
                    scale_factor=scale_factor
                )
                frame_dict[f'{former_idx}.{middle_i}'] = _middle_frame
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)


