# ComfyUI Frame Interpolation (ComfyUI VFI) (WIP)

A custom node set for Video Frame Interpolation in ComfyUI.

Big thanks for styler00dollar for making [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker). About 99% the code of this repo comes from it.

## Nodes
* KSampler Gradually Adding More Denoise (efficient)
* GMFSS Fortuna VFI
* IFRNet_VFI
* IFUnet VFI
* M2M VFI
* RIFE VFI (4.0 - 4.7) (Note that option `fast_mode` won't do anything from v4.5+ as `contextnet` is removed)
* FILM VFI
* Make Interpolation State List
* STMFNet VFI (requires at least 4 frames, can only do 2x interpolation for now)

## Install
### ComfyUI Manager
Incompatibile issue with it is now fixed

Following this guide to install this extension

https://github.com/ltdrdata/ComfyUI-Manager#how-to-use
### Command-line
#### Windows
Run install.bat

For Window users, if you are having trouble with cupy, please run `install.bat` instead of `install-cupy.py` or `python install.py`.
#### Linux
Open your shell app and start venv if it is used for ComfyUI. Run:
```
python install.py
```
## Support for non-CUDA device (experimental)
If you don't have a NVidia card, you can try `taichi` ops backend powered by [Taichi Lang](https://www.taichi-lang.org/)

On Windows, you can install it by running `install.bat` or `pip install taichi` on Linux

Then change value of `ops_backend` from `cupy` to `taichi` in `config.yaml`

If `NotImplementedError` appears, a VFI node in the workflow isn't supported by taichi

## Usage
All VFI nodes are placed in `ComfyUI-Frame-Interpolation/VFI` and require a `IMAGE` containing frames (at least 2, or at least 4 for STMF-Net).

Regarding STMFNet, if you only have two or three frames, you should use: Load Images -> Other VFI node (FILM is recommended in this case) with `multiplier=3` -> STMFNet VFI

The number of output frames is `N * multiplier - 1` for most of VFI models and STMFNet with `duplicate_first_last_frames` enabled.

`clear_cache_after_n_frames` is used to avoid out-of-memory. Decreasing it makes the chance lower but also increases processing time.

It is recommended to use LoadImages (LoadImagesFromDirectory) from [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/) and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) along side with this extension.

## Example
### Simple workflow
Workflow metadata isn't embeded
Download these two images [anime0.png](./demo_frames/anime0.png) and [anime1.png](./demo_frames/anime0.png) and put them into a folder like `E:\test` in this image.
![](./example.png)

### Complex workflow
It's used in AnimationDiff (can load workflow metadata)
![](All_in_one_v1_3.png)
