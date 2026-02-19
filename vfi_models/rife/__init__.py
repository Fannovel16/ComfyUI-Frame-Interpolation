import torch
from torch.utils.data import DataLoader
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
import typing
from comfy.model_management import get_torch_device
import re
from functools import cmp_to_key
from packaging import version

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
    #Arch 4.10 doesn't work due to state dict mismatch
    #TODO: Investigating and fix it
    #"rife410.pth": "4.10",
    #"rife411.pth": "4.10",
    #"rife412.pth": "4.10"
}

DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
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
                "fast_mode": ("BOOLEAN", {"default":True}),
                "ensemble": ("BOOLEAN", {"default":True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
                # Opt 1: Precision — float32 is safe, float16 is ~2x faster/half VRAM,
                #         bfloat16 is safer than float16 on Ampere+ GPUs (RTX 30xx+)
                "dtype": (DTYPE_OPTIONS, {"default": "float32"}),
                # Opt 2: torch.compile() — JIT-compiles the model graph for 10-30% speedup.
                #         First inference will be slow (compilation), subsequent ones faster.
                #         Requires PyTorch 2.0+. Disable if you hit errors on older builds.
                "torch_compile": ("BOOLEAN", {"default": False,
                                              "tooltip": "Compile the model with torch.compile() for 10-30% faster inference "
                                                         "after the first (warm-up) run. Requires PyTorch 2.0+."}),
                # Opt 7: Batch frame pairs — process multiple pairs per GPU call.
                #         Higher values improve GPU utilisation but use more VRAM.
                #         1 = original sequential behaviour (safest).
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Number of frame pairs to process per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Set to 1 to match original behaviour."}),
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
        fast_mode = False,
        ensemble = False,
        scale_factor = 1.0,
        dtype: str = "float32",
        torch_compile: bool = False,
        batch_size: int = 1,
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
            dtype (str, optional): Floating-point precision for inference.
                "float32" — full precision, safest (default).
                "float16" — half precision, ~2x faster and ~50% less VRAM, may have minor artifacts.
                "bfloat16" — brain float16, better numeric range than float16, requires Ampere+ GPU (RTX 30xx+).
            torch_compile (bool, optional): Compile the model with torch.compile() for 10-30% speedup.
                The first inference call will be slow (compilation warm-up); subsequent calls are faster.
                Requires PyTorch 2.0+. Defaults to False.
            batch_size (int, optional): Number of frame pairs to process per GPU call.
                Higher values improve GPU utilisation but consume more VRAM. Defaults to 1 (original behaviour).

        Returns:
            tuple: A tuple containing the output interpolated frames.

        Note:
            This method interpolates frames in a video sequence using a specified checkpoint model.
            It processes each frame sequentially, generating interpolated frames between them.

            To prevent memory overflow, it clears the CUDA cache after processing a specified number of frames.
        """
        from .rife_arch import IFNet
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        torch_dtype = DTYPE_MAP[dtype]
        interpolation_model = IFNet(arch_ver=arch_ver)
        interpolation_model.load_state_dict(torch.load(model_path))
        # Cast model weights to the chosen precision before moving to device
        if torch_dtype != torch.float32:
            interpolation_model = interpolation_model.to(torch_dtype)
        interpolation_model.eval().to(get_torch_device())
        # Opt 2: torch.compile() — JIT-compiles the model for 10-30% faster inference.
        # The first call triggers compilation (slow); all subsequent calls use the cached graph.
        if torch_compile:
            interpolation_model = torch.compile(interpolation_model)
        frames = preprocess_frames(frames)

        # Opt 5: wrap inference in inference_mode — disables autograd tracking entirely,
        #         faster and lower overhead than no_grad().
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
            with torch.inference_mode():
                return model(
                    frame_0.to(torch_dtype),
                    frame_1.to(torch_dtype),
                    timestep,
                    scale_list,
                    in_fast_mode,
                    in_ensemble
                )

        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        args = [interpolation_model, scale_list, fast_mode, ensemble]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args,
                               interpolation_states=optional_interpolation_states, dtype=torch_dtype,
                               batch_size=batch_size).to(torch.float32)
        )
        return (out,)
