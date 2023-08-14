import pathlib
from utils import load_file_from_github_release
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .eisai_arch import SoftsplatLite, DTM, RAFT

MODEL_TYPE = pathlib.Path(__file__).parent.name
MODEL_FILE_NAMES = {
    "ssl": "eisai_ssl.pt",
    "dtm": "eisai_dtm.pt",
    "raft": "eisai_anime_interp_full.ckpt"
}

class EISAI(nn.Module):
    def __init__(self, model_file_names) -> None:
        super(EISAI, self).__init__()
        self.raft = RAFT(load_file_from_github_release(MODEL_TYPE, model_file_names["raft"]))
        self.raft.cuda().eval()

        self.ssl = SoftsplatLite()
        self.ssl.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["ssl"])))
        self.ssl.cuda().eval()

        self.dtm = DTM()
        self.dtm.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["dtm"])))
        self.dtm.cuda().eval()
    
    def forward(self, img0, img1, t):
        with torch.no_grad():
            flow0, _ = self.raft(img0, img1)
            flow1, _ = self.raft(img1, img0)
            x = {
                "images": torch.stack([img0, img1], dim=1),
                "flows": torch.stack([flow0, flow1], dim=1),
            }
            out_ssl, _ = self.ssl(x, t=t, return_more=True)
            out_dtm, _ = self.dtm(x, out_ssl, _, return_more=False)
        return out_dtm[:, :3]

class EISAI_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["eisai"], ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000})
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
        optional_interpolation_states: typing.Optional[typing.List[bool]] = None
    ):
        global model
        model = EISAI(MODEL_FILE_NAMES)
        model.eval().cuda()
        frames = frames.cuda()
        frames = F.interpolate(frames, size=(540, 960)) #EISAI forces the input to be 960x540 lol

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
                    t=middle_i/multipler
                )
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        return torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)
