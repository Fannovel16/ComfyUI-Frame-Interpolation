import yaml
import os
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
import torch
import typing
import traceback
import einops
import torchvision.transforms.functional as transform
import numpy as np

BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/"
]
config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Frame-Interpolation")

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
    for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
        try:
            return load_file_from_url(base_model_download_url + ckpt_name, get_ckpt_container_path(model_type))
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no suceess. Below is the error log:\n\n{error_str}")
                

def load_file_from_direct_url(model_type, url):
    return load_file_from_url(url, get_ckpt_container_path(model_type))


def non_timestep_inference(model, I0, I1, multipler, **kwargs):
    """
    Return shape: (T, N, C, H, W), T = multipler - 1
    """
    batch_frames = [I0] + [None] * (multipler - 1) + [I1]
    middle_i = len(batch_frames) // 2
    batch_frames[middle_i] = model(I0, I1)

    for i in range(middle_i - 1, 0, -1):
        batch_frames[i] = model(batch_frames[0], batch_frames[i + 1], **kwargs)

    for i in range(middle_i + 1, len(batch_frames) - 1):
        batch_frames[i] = model(batch_frames[i - 1], batch_frames[-1], **kwargs)

    return torch.stack(batch_frames, dim=0)

def preprocess_frames(frames, device):
    return einops.rearrange(frames.to(device), "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c").cpu()

def generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.SupportsInt,
        return_middle_frame_function,
        *return_middle_frame_function_args):
    
    output_frames = []  # List to store processed frames in the correct order

    number_of_frames_processed_since_last_cleared_cuda_cache = 0
    
    for frame_itr in range(len(frames) - 1): # Skip the final frame since there are no frames after it
        frame_0 = frames[frame_itr]
        output_frames.append(frame_0) # Start with first frame
        
        # Generate and append a batch of middle frames
        middle_frames_batch = []

        # Generate and append a middle frame per multiplier - 1
        for middle_i in range(1, multiplier):
            timestep = middle_i/multiplier
            
            middle_frame = return_middle_frame_function(
                frame_0, 
                frames[frame_itr + 1],
                timestep,
                *return_middle_frame_function_args
            )
            middle_frames_batch.append(middle_frame)
            
        # Extend output array by batch
        output_frames.extend(middle_frames_batch)

        # Try to avoid a memory overflow by clearing cuda cache regularly
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            torch.cuda.empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
                
    output_frames.append(frames[-1]) # Append final frame
    out = torch.cat(output_frames, dim=0)
    # clear cache for courtesy
    torch.cuda.empty_cache()
    return out
    