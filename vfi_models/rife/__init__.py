import gc
import torch
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
from packaging import version

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
    # Arch 4.10 doesn't work due to state dict mismatch
    # "rife410.pth": "4.10",
    # "rife411.pth": "4.10",
    # "rife412.pth": "4.10"
}

DTYPE_OPTIONS = ["float32", "float16", "bfloat16"]
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Module-level model cache: avoids reloading weights on every node execution.
# Key: (ckpt_name, dtype, torch_compile) — invalidated if any of these change.
_model_cache: typing.Dict[typing.Tuple, torch.nn.Module] = {}


class RIFE_VFI:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name])),
                    {"default": "rife49.pth"}
                ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "fps_mode": ("BOOLEAN", {"default": False,
                                         "tooltip": "When enabled, use source_fps and target_fps to determine output frame count "
                                                    "instead of multiplier."}),
                "multiplier": ("INT", {"default": 2, "min": 1,
                                       "tooltip": "Used when fps_mode is off. Multiplies each input frame pair by this factor."}),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
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
                # Opt 7: Batch tasks — process multiple (pair, timestep) tasks per GPU call.
                #         Each task is one interpolated frame. Batching amortises kernel launch
                #         overhead; higher values improve throughput but use more VRAM.
                #         1 = one interpolated frame per call (safest).
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                       "tooltip": "Number of interpolation tasks per GPU call. "
                                                  "Higher values improve throughput but use more VRAM. "
                                                  "Set to 1 for the most conservative behaviour."}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
                "source_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.001,
                                         "tooltip": "Frame rate of the input frames. "
                                                    "Set both source_fps and target_fps (>0) to use FPS mode instead of multiplier."}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.001,
                                         "tooltip": "Desired output frame rate. "
                                                    "Set both source_fps and target_fps (>0) to use FPS mode instead of multiplier."}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames: int = 10,
        fps_mode: bool = False,
        multiplier: typing.SupportsInt = 2,
        fast_mode: bool = False,
        ensemble: bool = False,
        scale_factor: float = 1.0,
        dtype: str = "float32",
        torch_compile: bool = False,
        batch_size: int = 1,
        optional_interpolation_states: InterpolationStateList = None,
        source_fps: float = 0.0,
        target_fps: float = 0.0,
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
            batch_size (int, optional): Number of interpolation tasks per GPU call. Each task produces one
                intermediate frame. Higher values improve throughput but use more VRAM. Defaults to 1.

        Returns:
            tuple: A tuple containing the output interpolated frames.
        """
        from .rife_arch import IFNet

        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        torch_dtype = DTYPE_MAP[dtype]
        device = get_torch_device()

        # Cache the model by (ckpt_name, dtype, torch_compile) so repeated node
        # executions skip the load_state_dict + device transfer entirely.
        cache_key = (ckpt_name, dtype, torch_compile)
        if cache_key not in _model_cache:
            interpolation_model = IFNet(arch_ver=arch_ver)
            interpolation_model.load_state_dict(torch.load(model_path, weights_only=False))
            if torch_dtype != torch.float32:
                interpolation_model = interpolation_model.to(torch_dtype)
            interpolation_model.eval().to(device)
            # Opt 2: torch.compile() — JIT-compiles the model for 10-30% faster inference.
            # The first call triggers compilation (slow); all subsequent calls use the cached graph.
            if torch_compile:
                interpolation_model = torch.compile(interpolation_model)
            _model_cache[cache_key] = interpolation_model
            print(f"Comfy-VFI: Loaded and cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")
        else:
            interpolation_model = _model_cache[cache_key]
            print(f"Comfy-VFI: Using cached model {ckpt_name} ({dtype}{'+ torch.compile' if torch_compile else ''})")

        frames = preprocess_frames(frames)

        n_input = len(frames)
        n_pairs = n_input - 1
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        # output_specs: describes every output frame in order.
        #   ('orig', frame_idx)      — copy frames[frame_idx] directly
        #   ('interp', task_idx)     — result of tasks[task_idx]
        output_specs: typing.List[typing.Tuple] = []
        tasks: typing.List[typing.Tuple[int, float]] = []

        use_fps_mode = fps_mode and source_fps > 0 and target_fps > 0

        if use_fps_mode:
            # FPS mode: place output frames at exact target_fps timestamps.
            if abs(source_fps - target_fps) < 0.01:
                print("Comfy-VFI: source_fps ≈ target_fps, returning frames unchanged.")
                return (postprocess_frames(frames.to(torch.float32)),)

            n_output = max(2, round((n_input - 1) * target_fps / source_fps) + 1)
            print(f"Comfy-VFI: FPS mode {source_fps} → {target_fps} fps  ({n_input} → {n_output} frames)")

            for out_i in range(n_output):
                # Map output frame index to a continuous position in input-frame space
                input_pos = out_i * (n_input - 1) / (n_output - 1)
                pair_idx = int(input_pos)
                alpha = input_pos - pair_idx

                if pair_idx >= n_pairs or alpha < 1e-6:
                    # Exact input frame (or past the last)
                    output_specs.append(('orig', min(pair_idx, n_input - 1)))
                elif optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                    output_specs.append(('orig', pair_idx))
                else:
                    output_specs.append(('interp', len(tasks)))
                    tasks.append((pair_idx, alpha))

        else:
            # Multiplier mode: insert (multiplier-1) evenly-spaced frames between each pair.
            if isinstance(multiplier, int):
                multipliers = [int(multiplier)] * n_pairs
            else:
                multipliers = list(map(int, multiplier))
                multipliers += [2] * (n_pairs - len(multipliers))

            tasks_remaining_per_pair: typing.Dict[int, int] = {}
            for pair_idx in range(n_pairs):
                output_specs.append(('orig', pair_idx))
                if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                    tasks_remaining_per_pair[pair_idx] = 0
                    continue
                m = multipliers[pair_idx]
                n_steps = max(m - 1, 0)
                tasks_remaining_per_pair[pair_idx] = n_steps
                for step in range(1, m):
                    output_specs.append(('interp', len(tasks)))
                    tasks.append((pair_idx, step / m))
            output_specs.append(('orig', n_input - 1))

        # Flat array to hold each interpolated frame result, indexed by task position.
        interp_results: typing.List[typing.Optional[torch.Tensor]] = [None] * len(tasks)

        frames_processed_since_cache_clear = 0
        pos = 0

        # Opt 5: inference_mode disables autograd tracking entirely — lower overhead than no_grad().
        # Opt 7: process batch_size tasks per GPU call. IFNet supports batched tensor timesteps,
        #         so multiple (pair, timestep) combinations can be inferred in a single forward pass.
        with torch.inference_mode():
            while pos < len(tasks):
                batch_tasks = tasks[pos : pos + batch_size]

                frame0_list, frame1_list, timestep_list = [], [], []
                for pair_idx, dt in batch_tasks:
                    frame0_list.append(frames[pair_idx : pair_idx + 1])
                    frame1_list.append(frames[pair_idx + 1 : pair_idx + 2])
                    timestep_list.append(dt)

                frame0_batch = torch.cat(frame0_list, dim=0).to(device, dtype=torch_dtype)
                frame1_batch = torch.cat(frame1_list, dim=0).to(device, dtype=torch_dtype)
                # Batched timestep tensor: shape [B, 1, 1, 1] — IFNet expands it internally
                timestep_tensor = torch.tensor(timestep_list, dtype=torch_dtype, device=device).view(-1, 1, 1, 1)

                middle_frames = interpolation_model(
                    frame0_batch,
                    frame1_batch,
                    timestep_tensor,
                    scale_list,
                    fast_mode,
                    ensemble,
                ).clamp(0, 1).detach().cpu()

                for i, (pair_idx, _) in enumerate(batch_tasks):
                    task_idx = pos + i
                    interp_results[task_idx] = middle_frames[i : i + 1].to(dtype=torch_dtype)

                    if not use_fps_mode:
                        tasks_remaining_per_pair[pair_idx] -= 1
                        if tasks_remaining_per_pair[pair_idx] == 0:
                            frames_processed_since_cache_clear += 1
                            if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                                print("Comfy-VFI: Clearing cache...", end=' ')
                                soft_empty_cache()
                                gc.collect()
                                frames_processed_since_cache_clear = 0
                                print("Done cache clearing")

                pos += len(batch_tasks)

        # Assemble output frames in order using output_specs
        output_frames: typing.List[torch.Tensor] = []
        for spec in output_specs:
            if spec[0] == 'orig':
                output_frames.append(frames[spec[1] : spec[1] + 1].to(dtype=torch_dtype))
            else:
                output_frames.append(interp_results[spec[1]])

        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")
        print(f"Comfy-VFI done! {len(output_frames)} frames generated")

        # Always return float32 — numpy and all downstream ComfyUI nodes require it
        out_tensor = torch.cat(output_frames, dim=0).to(torch.float32)
        return (postprocess_frames(out_tensor),)
