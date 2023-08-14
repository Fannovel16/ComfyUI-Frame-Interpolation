import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release, preprocess_frames, postprocess_frames
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.1}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        batch_size: typing.SupportsInt = 1,
        multipler: typing.SupportsInt = 2,
        scale_factor: typing.SupportsFloat = 1.0,
        optional_interpolation_states: typing.Optional[list[bool]] = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        global model
        model = IRFNet_S() if 'S' in ckpt_name else IRFNet_L()
        model.load_state_dict(torch.load(model_path))
        model.eval().cuda()

        frames = preprocess_frames(frames, "cuda")
        
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
                    scale_factor=scale_factor
                )
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        out_frames = torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
        return (postprocess_frames(out_frames), )


