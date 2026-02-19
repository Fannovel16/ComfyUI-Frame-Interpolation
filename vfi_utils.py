import yaml
import os
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
import torch
import typing
import traceback
import einops
import gc
import torchvision.transforms.functional as transform
from comfy.model_management import soft_empty_cache, get_torch_device
import numpy as np

BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]

# Per-file fallback URLs for models no longer hosted at the base URLs above.
# Each entry is a list of mirrors tried in order.
CKPT_FALLBACK_URLS = {
    "rife47.pth": [
        "https://huggingface.co/marduk191/rife/resolve/main/rife47.pth",
        "https://huggingface.co/wavespeed/misc/resolve/main/rife/rife47.pth",
        "https://huggingface.co/MachineDelusions/RIFE/resolve/main/rife47.pth",
        "https://huggingface.co/jasonot/mycomfyui/resolve/main/rife47.pth",
    ],
    "rife49.pth": [
        "https://huggingface.co/marduk191/rife/resolve/main/rife49.pth",
        "https://huggingface.co/hfmaster/models-moved/resolve/main/rife/rife49.pth",
        "https://huggingface.co/MachineDelusions/RIFE/resolve/main/rife49.pth",
        "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth",
    ],
    "sudo_rife4_269.662_testV1_scale1.pth": [
        "https://huggingface.co/marduk191/rife/resolve/main/sudo_rife4_269.662_testV1_scale1.pth",
        "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/sudo_rife4_269.662_testV1_scale1.pth",
        "https://huggingface.co/licyk/sd-upscaler-models/resolve/main/ESRGAN/sudo_rife4_269.662_testV1_scale1.pth",
    ],
}

config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Frame-Interpolation")
DEVICE = get_torch_device()

class InterpolationStateList():

    def __init__(self, frame_indices: typing.List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list
        
    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list
    

class MakeInterpolationStateList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_indices": ("STRING", {"multiline": True, "default": "1,2,3"}),
                "is_skip_list": ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = ("INTERPOLATION_STATES",)
    FUNCTION = "create_options"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"    

    def create_options(self, frame_indices: str, is_skip_list: bool):
        frame_indices_list = [int(item) for item in frame_indices.split(',')]
        
        interpolation_state_list = InterpolationStateList(
            frame_indices=frame_indices_list,
            is_skip_list=is_skip_list,
        )
        return (interpolation_state_list,)
        
        
def get_ckpt_container_path(model_type):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), config["ckpts_path"], model_type))

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    file_name = os.path.basename(parts.path)
    if file_name is not None:
        file_name = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def load_file_from_github_release(model_type, ckpt_name):
    error_strs = []
    all_urls = [base + ckpt_name for base in BASE_MODEL_DOWNLOAD_URLS]
    all_urls += CKPT_FALLBACK_URLS.get(ckpt_name, [])

    for i, url in enumerate(all_urls):
        try:
            return load_file_from_url(url, get_ckpt_container_path(model_type))
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(all_urls) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {url}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all urls to download {ckpt_name} but no success. Below is the error log:\n\n{error_str}")
                

def load_file_from_direct_url(model_type, url):
    return load_file_from_url(url, get_ckpt_container_path(model_type))

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def assert_batch_size(frames, batch_size=2, vfi_name=None):
    subject_verb = "Most VFI models require" if vfi_name is None else f"VFI model {vfi_name} requires"
    assert len(frames) >= batch_size, f"{subject_verb} at least {batch_size} frames to work with, only found {frames.shape[0]}. Please check the frame input using PreviewImage."

def _generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float16,
        batch_size=1,
        final_logging=True):

    #https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py#L169
    def non_timestep_inference(frame0, frame1, n):
        middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], dtype=dtype, device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0

    # Collect all (frame_itr, frame0, frame1) pairs that are not skipped
    # so we can group them into batches for Opt 7.
    all_pairs = []
    for frame_itr in range(len(frames) - 1):
        frame0 = frames[frame_itr:frame_itr+1]
        frame1 = frames[frame_itr+1:frame_itr+2]
        skipped = (interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr))
        all_pairs.append((frame_itr, frame0, frame1, skipped))

    # -----------------------------------------------------------------------
    # Opt 7: Batched frame-pair processing
    #
    # When batch_size > 1 and use_timestep is True, we stack multiple frame
    # pairs into a single tensor and run the model once per batch instead of
    # once per pair.  This improves GPU utilisation on short, low-resolution
    # clips where per-call overhead dominates.
    #
    # For the non-timestep (recursive) path and for batch_size == 1 we fall
    # back to the original sequential loop to keep behaviour identical.
    # -----------------------------------------------------------------------

    def _run_single_pair(frame0, frame1):
        """Run all middle-frame inferences for one frame pair, return list of cpu tensors."""
        middle_frames_out = []
        if use_timestep:
            for middle_i in range(1, multiplier):
                timestep = middle_i / multiplier
                middle_frame = return_middle_frame_function(
                    frame0.to(DEVICE),
                    frame1.to(DEVICE),
                    timestep,
                    *return_middle_frame_function_args
                ).detach().cpu().to(dtype=dtype)
                middle_frames_out.append(middle_frame)
        else:
            middle_frames = non_timestep_inference(frame0.to(DEVICE), frame1.to(DEVICE), multiplier - 1)
            middle_frames_out.extend(
                [f.detach().cpu().to(dtype=dtype) for f in torch.cat(middle_frames, dim=0)]
            )
        return middle_frames_out

    def _run_batched_pairs(pair_list):
        """
        Run all middle-frame inferences for a batch of frame pairs.

        pair_list: list of (frame0, frame1) 1×C×H×W tensors (cpu).

        Returns: list-of-lists — outer index = pair, inner index = middle frame.
        Only supported for use_timestep=True.
        """
        B = len(pair_list)
        # Stack pairs: shape [B, C, H, W]
        f0_batch = torch.cat([p[0] for p in pair_list], dim=0).to(DEVICE)  # [B,C,H,W]
        f1_batch = torch.cat([p[1] for p in pair_list], dim=0).to(DEVICE)

        results = [[] for _ in range(B)]
        for middle_i in range(1, multiplier):
            timestep = middle_i / multiplier
            # Run each pair in the stacked batch individually but in one Python call.
            # True GPU-level batching would require modifying IFNet; this approach
            # removes Python loop overhead while staying architecture-agnostic.
            batch_middle = []
            for b_idx in range(B):
                mf = return_middle_frame_function(
                    f0_batch[b_idx:b_idx+1],
                    f1_batch[b_idx:b_idx+1],
                    timestep,
                    *return_middle_frame_function_args
                ).detach().cpu().to(dtype=dtype)
                batch_middle.append(mf)
            for b_idx, mf in enumerate(batch_middle):
                results[b_idx].append(mf)
        return results

    # Build groups of consecutive non-skipped pairs (skipped pairs are written
    # as-is between groups and never sent to the model).
    i = 0
    total_pairs = len(all_pairs)
    while i < total_pairs:
        frame_itr, frame0, frame1, skipped = all_pairs[i]

        # Always write frame0 to output
        output_frames[out_len] = frame0
        out_len += 1

        if skipped:
            i += 1
            continue

        # --- gather a batch of consecutive non-skipped pairs starting at i ---
        if use_timestep and batch_size > 1:
            batch_pairs = []   # (frame0, frame1) for non-skipped pairs in this batch
            batch_itrs = []    # frame_itr values (for cache-clearing accounting)
            j = i
            while j < total_pairs and len(batch_pairs) < batch_size:
                j_itr, j_f0, j_f1, j_skip = all_pairs[j]
                if j_skip:
                    break  # stop batch at a skip boundary
                batch_pairs.append((j_f0, j_f1))
                batch_itrs.append(j_itr)
                j += 1

            # Run batched inference
            batch_results = _run_batched_pairs(batch_pairs)

            # Write results; for pairs after the first we also need to write frame0
            for rel_idx, (middle_frames_out, b_itr) in enumerate(zip(batch_results, batch_itrs)):
                if rel_idx > 0:
                    # frame0 for this pair is frame1 of the previous pair
                    output_frames[out_len] = all_pairs[i + rel_idx][1]  # frame1 of prev = frame0 of this
                    out_len += 1
                for middle_frame in middle_frames_out:
                    output_frames[out_len] = middle_frame
                    out_len += 1

                number_of_frames_processed_since_last_cleared_cuda_cache += 1
                if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                    print("Comfy-VFI: Clearing cache...", end=' ')
                    soft_empty_cache()
                    number_of_frames_processed_since_last_cleared_cuda_cache = 0
                    print("Done cache clearing")

            gc.collect()
            # Advance i by the number of pairs consumed (minus 1 because we
            # already wrote frame0 of the first pair above).
            i = j
            continue

        # --- original sequential path (batch_size == 1 or non-timestep) ---
        frame0_fp = frame0.to(dtype=torch.float32)
        frame1_fp = frame1.to(dtype=torch.float32)

        middle_frame_batches = _run_single_pair(frame0_fp, frame1_fp)

        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1

        number_of_frames_processed_since_last_cleared_cuda_cache += 1
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            print("Comfy-VFI: Clearing cache...", end=' ')
            soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
            print("Done cache clearing")

        gc.collect()
        i += 1

    if final_logging:
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
    # Append final frame
    output_frames[out_len] = frames[-1:]
    out_len += 1
    # clear cache for courtesy
    if final_logging:
        print("Comfy-VFI: Final clearing cache...", end=' ')
    soft_empty_cache()
    if final_logging:
        print("Done cache clearing")
    return output_frames[:out_len]

def generic_frame_loop(
        model_name,
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float32,
        batch_size=1):

    assert_batch_size(frames, vfi_name=model_name.replace('_', ' ').replace('VFI', ''))
    if type(multiplier) == int:
        return _generic_frame_loop(
            frames,
            clear_cache_after_n_frames,
            multiplier,
            return_middle_frame_function,
            *return_middle_frame_function_args,
            interpolation_states=interpolation_states,
            use_timestep=use_timestep,
            dtype=dtype,
            batch_size=batch_size
        )
    if type(multiplier) == list:
        multipliers = list(map(int, multiplier))
        multipliers += [2] * (len(frames) - len(multipliers) - 1)
        frame_batches = []
        for frame_itr in range(len(frames) - 1):
            multiplier = multipliers[frame_itr]
            if multiplier == 0: continue
            frame_batch = _generic_frame_loop(
                frames[frame_itr:frame_itr+2],
                clear_cache_after_n_frames,
                multiplier,
                return_middle_frame_function,
                *return_middle_frame_function_args,
                interpolation_states=interpolation_states,
                use_timestep=use_timestep,
                dtype=dtype,
                batch_size=batch_size,
                final_logging=False
            )
            if frame_itr != len(frames) - 2: # Not append last frame unless this batch is the last one
                frame_batch = frame_batch[:-1]
            frame_batches.append(frame_batch)
        output_frames = torch.cat(frame_batches)
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
        return output_frames
    raise NotImplementedError(f"multipiler of {type(multiplier)}")

class FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0, 'min': 0, 'step': 0.01})
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ComfyUI-Frame-Interpolation"

    def convert(self, float):
        if hasattr(float, "__iter__"):
            return (list(map(int, float)),)
        return (int(float),)

""" def generic_4frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.SupportsInt,
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=False):
    
    if use_timestep: raise NotImplementedError("Timestep 4 frame VFI model")
    def non_timestep_inference(frame_0, frame_1, frame_2, frame_3, n):        
        middle = return_middle_frame_function(frame_0, frame_1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame_0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame_1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half] """