import torch
import torch.nn.functional as F
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames
import pathlib
import gc

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode='replicate')
        else:
            return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]
    
    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["atm-vfi-base.pt", "atm-vfi-lite.pt", "atm-vfi-base-pct.pt"]
DEVICE = get_torch_device()
def inference(model, img_batch_1, img_batch_2, inter_frames):
    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i].to(DEVICE)
        x1 = results[end_i].to(DEVICE)

        # border padding
        padder = InputPadder(x0.shape, divisor=64)
        x0, x1 = padder.pad(x0, x1)

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = padder.unpad(model.forward(x0, x1)["I_t"][0]).unsqueeze(0)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]

global_motion_settings = {
    "On": [True, False],
    "On with Ensemble (slowest)": [True, True],
    "Off (fastest)": [False, False]
}


class ATM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 2}),
                "global_motion": (["On", "On with Ensemble (slowest)", "Off (fastest)"],),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"        

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        global_motion = "On",
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        interpolation_states = optional_interpolation_states

        if ckpt_name == "atm-vfi-lite.pt":
            from .network_lite import Network
        else:
            from .network_base import Network
        
        model = Network()

        settings = global_motion_settings.get(global_motion)

        model.global_motion = settings[0]

        model.ensemble_global_motion = settings[1]

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        device = get_torch_device()
        checkpoint = torch.load(model_path, map_location='cpu')
        param = checkpoint['model_state_dict']

        layers_to_remove = []
        for key in param:
            if "attn_mask" in key:
                layers_to_remove.append(key)
            elif "HW" in key:
                layers_to_remove.append(key)
		
        for key in layers_to_remove:
            del param[key]
        model.load_state_dict(param, strict=True)

        model = model.to(DEVICE)
        dtype = torch.float32


        frames = preprocess_frames(frames)
        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        
        if type(multiplier) == int:
            multipliers = [multiplier] * len(frames)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (len(frames) - len(multipliers) - 1)
        for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue
            #Ensure that input frames are in fp32 - the same dtype as model
            frame_0 = frames[frame_itr:frame_itr+1].to(DEVICE).float()
            frame_1 = frames[frame_itr+1:frame_itr+2].to(DEVICE).float()
            relust = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1)
            output_frames.extend([frame.detach().cpu().to(dtype=dtype) for frame in relust[:-1]])

            number_of_frames_processed_since_last_cleared_cuda_cache += 1
            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end = ' ')
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")
            gc.collect()
        
        output_frames.append(frames[-1:].to(dtype=dtype)) # Append final frame
        output_frames = [frame.cpu() for frame in output_frames] #Ensure all frames are in cpu
        out = torch.cat(output_frames, dim=0)
        # clear cache for courtesy
        print("Comfy-VFI: Final clearing cache...", end = ' ')
        soft_empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )
