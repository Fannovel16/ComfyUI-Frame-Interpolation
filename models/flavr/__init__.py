import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import einops
import pathlib
import typing
from .flavr_arch import UNet_3D_3D
from utils import load_file_from_github_release, preprocess_frames

NBR_FRAME = 4

class FLAVR_Inference(nn.Module):
    def __init__(self, model_path) -> None:
        super(FLAVR_Inference, self).__init__()
        sd = torch.load(model_path)['state_dict']
        sd = {k.partition("module.")[-1]:v for k,v in sd.items()}

        #Ref: Class UNet_3D_3D
        self.model = UNet_3D_3D("unet_18", n_inputs=NBR_FRAME, n_outputs=sd["outconv.1.weight"].shape[0] // 3, joinType="concat" , upmode="transpose")
        self.model.load_state_dict(sd)
        self.model.cuda().eval()
        del sd
    
    def forward(self, frame_tensor):
        """
        Expect a Tensor of size 4CHW.

        Ref: https://github.com/tarun005/FLAVR/blob/main/interpolate.py

        Why the hell the author transposes the tensor THWC -> CTHW -> CHW -> 1CHW?

        Why don't just use TCHW?
        """
        frame_amount = len(frame_tensor)
        if frame_amount != NBR_FRAME:
            raise RuntimeError(f"FLAVR FVI model requires {NBR_FRAME} frames to work with (found {frame_amount}).")

        outputs = self.model([frame.unsqueeze(0) for frame in torch.unbind(frame_tensor, dim=0)])
        outputs.append(frame_tensor[2].unsqueeze(0))

        return torch.cat(outputs, dim=0)


MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["FLAVR_2x.pth", "FLAVR_4x.pth", "FLAVR_8x.pth"]


class FLAVR_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", )
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
        optional_interpolation_states: typing.Optional[list[bool]] = None
    ):
        if len(frames) < NBR_FRAME:
            raise RuntimeError(f"FLAVR requires at least {NBR_FRAME} frames to work with (found {len(frames)}).")

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        global model
        model = FLAVR_Inference(model_path)

        if optional_interpolation_states is None:
            interpolation_states = [True] * (frames.shape[0] - 1)
        else:
            interpolation_states = optional_interpolation_states

        enabled_former_idxs = [i for i, state in enumerate(interpolation_states) if state]
        frame_idx_batches = torch.tensor(range(len(frames))).type(torch.long).view(1,-1).unfold(1,size=NBR_FRAME,step=1).squeeze(0)
        """
        Example: tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
        """
        #Ref: https://github.com/tarun005/FLAVR/blob/main/interpolate.py#L146
        #Someone explains how this index batch thing works plz

        out_frames = [frames[frame_idx_batches[0][1]]]
        for frame_idx_batch in frame_idx_batches:
            if (frame_idx_batch[0] in enabled_former_idxs) or (frame_idx_batch[2] in enabled_former_idxs):
                print(frames[frame_idx_batch].shape)
                out_frames.extend(model(frames[frame_idx_batch]))
            else:
                #Dunno if this line is right lol
                out_frames.extend([frames[frame_idx_batch[0]].unsqueeze(0), frames[frame_idx_batch[1]].unsqueeze(0)])

        return (torch.stack(out_frames), )
