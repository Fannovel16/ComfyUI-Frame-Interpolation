import pathlib
from utils import load_file_from_github_release, preprocess_frames, postprocess_frames
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .eisai_arch import SoftsplatLite, DTM, RAFT
from comfy.model_management import soft_empty_cache, get_torch_device

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
        self.raft.to(get_torch_device()).eval()

        self.ssl = SoftsplatLite()
        self.ssl.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["ssl"])))
        self.ssl.to(get_torch_device()).eval()

        self.dtm = DTM()
        self.dtm.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["dtm"])))
        self.dtm.to(get_torch_device()).eval()
    
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

class EISAI_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {""}}

class EISAI_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("EISAI_MODEL", ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    @classmethod
    def CKPT_NAMES(s):
        return (["eisai"], {"default": EISAI})

    @classmethod
    def create_model(self, ckpt_name):
        model = EISAI(MODEL_FILE_NAMES)
        model.eval().to(get_torch_device())
        return model

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        model,
        orig_frames: torch.Tensor,
        batch_size: typing.SupportsInt = 1,
        multipler: typing.SupportsInt = 2,
        optional_interpolation_states: typing.Optional[typing.List[bool]] = None
    ):
        orig_frames = preprocess_frames(orig_frames, get_torch_device())
        orig_frames = F.interpolate(orig_frames, size=(540, 960)) #EISAI forces the input to be 960x540 lol
        frames_2d = [
            [orig_frame.unsqueeze(0)] for orig_frame in orig_frames
        ] 

        if optional_interpolation_states is None:
            interpolation_states = [True] * (orig_frames.shape[0] - 1)
        else:
            interpolation_states = optional_interpolation_states

        enabled_former_idxs = [i for i, state in enumerate(interpolation_states) if state]
        former_idxs_batches = [enabled_former_idxs[i:i + batch_size] for i in range(0, len(enabled_former_idxs), batch_size)] 

        for former_idxs_batch in former_idxs_batches:
            for middle_idx in range(1, multipler):
                middle_frames = model(
                    orig_frames[former_idxs_batch], 
                    orig_frames[former_idxs_batch + 1], 
                    t=middle_idx/multipler
                )
                
                for _idx, former_idx in enumerate(former_idxs_batch):
                    frames_2d[former_idx].append(middle_frames[_idx].unsqueeze(0))

        out_frames = torch.cat([torch.cat(frames) for frames in frames_2d], dim=0)
        return (postprocess_frames(out_frames), )
