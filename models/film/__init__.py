import torch
from comfy.model_management import get_torch_device
import bisect
import numpy as np
import typing
from utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames, soft_empty_cache
import pathlib

MODEL_TYPE = pathlib.Path(__file__).parent.name
device = get_torch_device()
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

        x0 = results[start_i].to(device)
        x1 = results[end_i].to(device)
        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).float())
        del remains[step]

    return [tensor.flip(0) for tensor in results]

class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
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
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None   
    ):
        interpolation_states = optional_interpolation_states
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        model = model.to(device)

        frames = preprocess_frames(frames, device)
        # Ensure proper tensor dimensions
        frames = [frame.unsqueeze(0) for frame in frames]

        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it
            frame_0 = frames[frame_itr]
            
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                continue
            
            relust = inference(model, frame_0, frames[frame_itr + 1], multiplier - 1)
            output_frames.extend(relust[:-1])

            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
        
        output_frames.append(frames[-1]) # Append final frame
        out = torch.cat(output_frames, dim=0)
        # clear cache for courtesy
        soft_empty_cache()
        return (postprocess_frames(out), )
