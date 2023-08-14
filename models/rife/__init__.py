from .rife_arch import IFNet
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release
import typing

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0", 
    "rife42.pth": "4.2", 
    "rife43.pth": "4.3", 
    "rife44.pth": "4.3", 
    "rife45.pth": "4.5",
    "rife46.pth": "4.6", 
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
}

class RIFE_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_NAME_VER_DICT.keys()), ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 1}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
                "fast_mode": (["enabled", "disabled"], ),
                "ensemble": (["enabled", "disabled"], )
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGES", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        batch_size: typing.SupportsInt = 1,
        multipler: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        fast_mode: typing.AnyStr = "enabled",
        ensemble: typing.AnyStr = "disabled",
        optional_interpolation_states: typing.Optional[list[bool]] = None
    ):
        fast_mode = fast_mode == "enabled"
        ensemble = ensemble == "enabled"
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        global model
        model = IFNet(arch_ver=CKPT_NAME_VER_DICT[ckpt_name])
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
        former_idxs_loader = DataLoader(enabled_former_idxs, batch_size=batch_size)
        
        for former_idxs_batch in former_idxs_loader:
            for middle_i in range(1, multipler):
                _middle_frames = model(
                    frames[former_idxs_batch], 
                    frames[former_idxs_batch + 1], 
                    timestep=middle_i/multipler,
                    scale_list=scale_list,
                    fastmode=fast_mode,
                    ensemble=ensemble
                )
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
