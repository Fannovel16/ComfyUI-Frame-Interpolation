import pathlib
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_direct_url
import typing
from .amt_arch import AMT_S, AMT_L, AMT_G, InputPadder

#https://github.com/MCG-NKU/AMT/tree/main/cfgs
CKPT_CONFIGS = {
    "amt-s.pth": {
        "network": AMT_S,
        "params": { "corr_radius": 3, "corr_lvls": 4, "num_flows": 3 }
    },
    "amt-l.pth": {
        "network": AMT_L,
        "params": { "corr_radius": 3, "corr_lvls": 4, "num_flows": 5 }
    },
    "amt-g.pth": {
        "network": AMT_G,
        "params": { "corr_radius": 3, "corr_lvls": 4, "num_flows": 5 }
    },
    "gopro_amt-s.pth": {
        "network": AMT_S,
        "params": { "corr_radius": 3, "corr_lvls": 4, "num_flows": 3 }
    }
}



MODEL_TYPE = pathlib.Path(__file__).parent.name

class AMT_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_CONFIGS.keys()), ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 1}),
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
        model_path = load_file_from_direct_url(MODEL_TYPE, f"https://huggingface.co/lalala125/AMT/resolve/main/{ckpt_name}")
        ckpt_config = CKPT_CONFIGS[ckpt_name]

        global model
        model = ckpt_config["network"](**ckpt_config["params"])
        model.load_state_dict(torch.load(model_path)["state_dict"])
        model.eval().cuda()

        frames.cuda()
        padder = InputPadder(frames[0].shape, 16)
        frames = torch.cat(padder.pad(*[frame.unsqueeze(0) for frame in frames]), dim=0)
        
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
                shape = frames[former_idxs_batch].shape
                _middle_frames = model(
                    frames[former_idxs_batch], 
                    frames[former_idxs_batch + 1], 
                    embt=torch.FloatTensor([middle_i / multipler] * shape[0]).view(shape[0], 1, 1, 1).cuda(),
                    scale_factor=scale_factor,
                    eval=True
                )["imgt_pred"]
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
