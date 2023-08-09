from .sepconv_enhanced import Network
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release
import typing

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["sepconv.pth"]


class SepconvVFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    
    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        batch_size: typing.SupportsInt = 1,
        optional_interpolation_states: typing.Optional[list[bool]] = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        global model
        model = Network()
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
            _middle_frames = model(frames[former_idxs_batch], frames[former_idxs_batch + 1])
            for i, former_idx in enumerate(former_idxs_batch):
                frame_dict[f'{former_idx}.0'] = _middle_frames[i].unsqueeze(0)
        
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
