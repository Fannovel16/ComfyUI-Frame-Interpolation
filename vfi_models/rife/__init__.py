import torch
from torch.utils.data import DataLoader
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
import re
from functools import cmp_to_key
from packaging import version
import gc

MODEL_TYPE = pathlib.Path(__file__).parent.name

CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0",
    "rife42.pth": "4.2",
    "rife43.pth": "4.3",
    "rife44.pth": "4.3",
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
    # Arch 4.10 doesn't work due to state dict mismatch
    # "rife410.pth": "4.10",
    # "rife411.pth": "4.10",
    # "rife412.pth": "4.10"
}


class RIFE_VFI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife47.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    # ComfyUI uses RETURN_TYPES and FUNCTION to determine how to wire nodes
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        fast_mode = False,
        ensemble = False,
        scale_factor = 1.0,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        """
        Perform video frame interpolation using a given checkpoint model.
    
        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
    
        Returns:
            tuple: A tuple containing the output interpolated frames.
    
        Note:
            This method interpolates frames in a video sequence using a specified checkpoint model. 
            It processes each frame sequentially, generating interpolated frames between them.
    
            To prevent memory overflow, it clears the CUDA cache after processing a specified number of frames.
        """
        
        # Local import of the model definition to avoid circular imports
        from .rife_arch import IFNet

        # Resolve the checkpoint path and instantiate the model
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        interpolation_model = IFNet(arch_ver=arch_ver)
        interpolation_model.load_state_dict(torch.load(model_path))

        # Move model to correct device and set to eval mode
        device = get_torch_device()
        interpolation_model.eval().to(device)

        # Convert input frames from NHWC to NCHW and ensure float32 dtype
        frames = preprocess_frames(frames)
        dtype = torch.float32

        # Prepare per-frame multipliers
        if isinstance(multiplier, int):
            multipliers = [int(multiplier)] * len(frames)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (len(frames) - len(multipliers))

        # Determine the scale list used by RIFE for multi-scale processing
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        output_frames: typing.List[torch.Tensor] = []
        frames_processed_since_cache_clear = 0

        # Build a list of interpolation tasks across all frame pairs. Each task is a
        # tuple of (pair_idx, dt) representing the pair index and timestep fraction.
        # Pairs that are skipped via optional_interpolation_states have no tasks.
        tasks: typing.List[typing.Tuple[int, float]] = []
        num_tasks_per_pair: typing.Dict[int, int] = {}
        for pair_idx in range(len(frames) - 1):
            if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                num_tasks_per_pair[pair_idx] = 0
                continue
            m = multipliers[pair_idx]
            n = max(m - 1, 0)
            num_tasks_per_pair[pair_idx] = n
            for step in range(1, m):
                tasks.append((pair_idx, step / m))

        # Dictionary mapping pair index to list of intermediate frames
        results: typing.Dict[int, typing.List[torch.Tensor]] = {i: [] for i in range(len(frames) - 1)}

        pos = 0
        while pos < len(tasks):
            # Always process a single task at a time since batching is disabled.
            batch_tasks = tasks[pos : pos + 1]
            # prepare lists
            frame0_list: typing.List[torch.Tensor] = []
            frame1_list: typing.List[torch.Tensor] = []
            timestep_list: typing.List[float] = []
            for (pair_idx, dt) in batch_tasks:
                frame0_cpu = frames[pair_idx:pair_idx+1]
                frame1_cpu = frames[pair_idx+1:pair_idx+2]
                frame0_list.append(frame0_cpu)
                frame1_list.append(frame1_cpu)
                timestep_list.append(dt)
            # combine and move to device
            frame0_batch = torch.cat(frame0_list, dim=0).to(device).to(dtype)
            frame1_batch = torch.cat(frame1_list, dim=0).to(device).to(dtype)
            timestep_tensor = torch.tensor(timestep_list, dtype=dtype, device=device).view(-1, 1, 1, 1)

            with torch.no_grad():
                middle_frames = interpolation_model(
                    frame0_batch,
                    frame1_batch,
                    timestep_tensor,
                    scale_list,
                    fast_mode,
                    ensemble
                ).clamp(0, 1)

            middle_frames_cpu = middle_frames.detach().cpu().to(dtype)

            # assign outputs
            for idx, (pair_idx, _dt) in enumerate(batch_tasks):
                results[pair_idx].append(middle_frames_cpu[idx:idx+1])
                num_tasks_per_pair[pair_idx] -= 1
                if num_tasks_per_pair[pair_idx] == 0:
                    frames_processed_since_cache_clear += 1
                    if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                        print("Comfy-VFI: Clearing cache...", end=' ')
                        soft_empty_cache()
                        frames_processed_since_cache_clear = 0
                        print("Done cache clearing")
                        gc.collect()
            pos += 1

        # Assemble the final output: original frames with their interpolated frames
        for frame_idx in range(len(frames) - 1):
            frame0_cpu = frames[frame_idx:frame_idx+1]
            output_frames.append(frame0_cpu.to(dtype=dtype))
            # append intermediate frames if pair not skipped
            if optional_interpolation_states is None or not optional_interpolation_states.is_frame_skipped(frame_idx):
                for mid in results[frame_idx]:
                    output_frames.append(mid)
        # append last frame
        output_frames.append(frames[-1:].to(dtype=dtype))

        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")

        out_tensor = torch.cat(output_frames, dim=0)
        out_images = postprocess_frames(out_tensor)
        return (out_images,)
