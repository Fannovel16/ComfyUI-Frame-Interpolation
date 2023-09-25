from .rife_arch import IFNet
import torch
from torch.utils.data import DataLoader
import pathlib
from utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop
import typing

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0", 
    "rife42.pth": "4.2", 
    "rife43.pth": "4.3", 
    "rife44.pth": "4.3", 
    "rife45.pth": "4.5",
    "rife46.pth": "4.6", 
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
}

class RIFE_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_NAME_VER_DICT.keys()), ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 1}),
                "fast_mode": ("BOOLEAN", {"default":False}),
                "ensemble": ("BOOLEAN", {"default":False}),
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
        fast_mode = False,
        ensemble = False,
        optional_interpolation_states: typing.Optional[list[bool]] = None    
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
        
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        interpolation_model = IFNet(arch_ver=CKPT_NAME_VER_DICT[ckpt_name])
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().cuda()
        
        frames = preprocess_frames(frames, "cuda")
            
        # Ensure proper tensor dimensions
        frames = [frame.unsqueeze(0) for frame in frames]
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
            return model(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)
        
        scale_factor = 1
        # Intentionally leaving this in, as scale_factor could become a parameter
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor] 
        
        args = [interpolation_model, scale_list, fast_mode, ensemble]
        out = postprocess_frames(
            generic_frame_loop(frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states)
        )
        return (out,)
