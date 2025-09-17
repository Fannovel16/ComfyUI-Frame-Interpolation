import torch
import torch.nn.functional as F
import pathlib
import typing
import os
import warnings

# --- ComfyUI-Frame-Interpolation Imports ---
try:
    # Use .. to navigate up one directory from 'practicalrife' to 'ComfyUI-Frame-Interpolation'
    from ..vfi_utils import preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
except ImportError:
    print("Attempting fallback import for vfi_utils...")
    try:
        from vfi_utils import preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
    except ImportError:
        raise ImportError("Could not import vfi_utils. Make sure it's accessible.")

from comfy.model_management import get_torch_device, VRAMState, soft_empty_cache

# --- Practical RIFE Model Import ---
# Assuming RIFE_HDv3.py is in a 'train_log' subdirectory relative to this file
try:
    from .train_log.RIFE_HDv3 import Model as RIFEv3Model
    RIFE_Model_Class = RIFEv3Model
    print("Successfully imported Practical RIFE Model class.")
except ImportError as e:
    print(f"Error importing Practical RIFE Model class: {e}")
    print("Ensure RIFE_HDv3.py (and dependencies) are in the 'train_log' subdirectory.")
    RIFE_Model_Class = None

# --- Node Definition ---

class PracticalRIFE_VFI:
    @classmethod
    def INPUT_TYPES(cls):
        if RIFE_Model_Class is None:
             return { "required": { "error": ("STRING", {"default": "Practical RIFE Model class failed to import. Check console.", "multiline": True}) } }

        return {
            "required": {
                "model_path": ("STRING", {"default": "ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/vfi_models/practicalrife/train_log", "multiline": False}),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 16}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
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
        model_path: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames: int = 10,
        gpu_id: int = 0,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        if multiplier == 1:
            return (frames,)

        if RIFE_Model_Class is None:
            raise ImportError("Practical RIFE Model class was not imported successfully. Cannot proceed.")

        # --- Device Setup ---
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")

        # --- Model Loading ---
        # Ensure model_path exists and is a directory
        if not os.path.isdir(model_path):
             raise NotADirectoryError(f"Practical RIFE model path is not a directory: {model_path}")

        interpolation_model = RIFE_Model_Class()
        try:
            interpolation_model.load_model(model_path, -1) # Assuming -1 handles device selection internally or loads to CPU
            print(f"Loaded Practical RIFE model (Version: {getattr(interpolation_model, 'version', 'Unknown')}) from: {model_path}")
        except Exception as e:
            print(f"Error loading Practical RIFE model: {e}")
            raise RuntimeError(f"Failed to load Practical RIFE model from {model_path}. Check console for details.")

        # --- Model Preparation ---
        interpolation_model.eval()
        try:
            interpolation_model.flownet.to(device)
            print(f"Model's flownet component moved to device: {device}")
        except AttributeError:
            print(f"Warning: Could not directly access 'flownet'. Moving the entire interpolation_model to {device}.")
            interpolation_model.to(device)

        # --- Frame Preparation ---
        # 1. Preprocess (NHWC -> NCHW) using vfi_utils function
        print("Preprocessing frames (NHWC -> NCHW)...")
        preprocessed_frames_chw = preprocess_frames(frames) # Result is NCHW, likely uint8

        # 2. Normalize (0-255 -> 0-1) - Assuming input frames are 8-bit
        print("Normalizing frames (uint8 -> float32, 0-1)...")
        # Convert to float before dividing
        preprocessed_frames_norm = preprocessed_frames_chw.float() #/ 255.0

        # 3. Move normalized frames to the target device (needed for padding)
        preprocessed_frames_device = preprocessed_frames_norm.to(device)
        print(f"Normalized frames moved to device: {device}")

        # 4. Get shape from the tensor on the correct device
        num_frames, c, h, w = preprocessed_frames_device.shape

        # 5. Pad frames to be divisible by --64-- seems to be 128 actually (required by Practical RIFE)
        # TODO: this doesn't actually work and needs to be fixed
        pad_h = ((h - 1) // 128 + 1) * 128
        pad_w = ((w - 1) // 128 + 1) * 128
        padding = (0, pad_w - w, 0, pad_h - h)
        # Padding needs to happen on the device the tensor is on
        padded_frames = F.pad(preprocessed_frames_device, padding, mode='replicate')
        print(f"Input frames {h}x{w} padded to {pad_h}x{pad_w} on device {device}")

        # --- Define Inference Function for generic_frame_loop ---
        def return_middle_frame(frame_0, frame_1, timestep, model_instance):
            with torch.no_grad():
                # Use timestep= parameter name
                out_frame = model_instance.inference(frame_0, frame_1, timestep=float(timestep))

            # The model outputs a batched frame [1, C, H, W].
            # Squeeze it to [C, H, W] for the loop to handle.
            return out_frame.squeeze(0)
            
        # --- Run Interpolation ---
        print("Starting generic frame loop...")
        args = [interpolation_model]
        # generic_frame_loop returns float32 tensor on CPU
        all_interpolated_frames_cpu = generic_frame_loop(type(self).__name__, padded_frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, interpolation_states=optional_interpolation_states, dtype=torch.float32)
        print("Generic frame loop finished.")

        # --- Postprocess ---
        # 1. Unpad frames to their original dimensions
        unpadded_frames_cpu_float = all_interpolated_frames_cpu[..., :h, :w]
        print(f"Output frames unpadded back to {h}x{w} (on CPU, float32)")

        # 2. Permute channels from NCHW to NHWC using the utility. The tensor remains float32.
        print("Postprocessing frames (NCHW -> NHWC)...")
        postprocessed_frames_float = postprocess_frames(unpadded_frames_cpu_float)

        # 3. Manually denormalize: scale values from [0.0, 1.0] to [0, 255] and convert to uint8.
        print("Denormalizing frames (float32 -> uint8)...")
        output_frames = postprocessed_frames_float #* 255.0
        print(f"Final output frames: {output_frames.shape}, {output_frames.dtype}, {output_frames.device}")

        # --- Cleanup ---
        soft_empty_cache()
        print("Cache cleared.")

        return (output_frames,)
        
# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "PracticalRIFE_VFI": PracticalRIFE_VFI
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PracticalRIFE_VFI": "Practical RIFE VFI"
}
