# ComfyUI Frame Interpolation (ComfyUI VFI) (WIP)

A custom node set for Video Frame Interpolation in ComfyUI.

Big thanks for styler00dollar for making [VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker). About 99% the code of this repo comes from it.

## Nodes
* KSampler Gradually Adding More Denoise (efficient)
* GMFSS Fortuna VFI
* IFRNet_VFI
* IFUnet VFI
* M2M VFI
* RIFE VFI (Note that option `fast_mode` won't do anything from v4.5+ as `contextnet` is removed)
* FILM VFI
* Make Interpolation State List

## Install
### ComfyUI Manager
https://github.com/ltdrdata/ComfyUI-Manager#how-to-use
### Command-line
#### Windows
Run install.bat

For Window users, if you are having trouble with cupy, please run `install.bat` instead of `install-cupy.py` as only the former can detect ComfyUI portable.
#### Linux
Open your shell app and start venv if it is used for ComfyUI. Then `cd` to this extension's directory and run
```
./install
```
## Support for non-CUDA device (experimental)
If you don't have a NVidia card, you can try `taichi` ops backend powered by [Taichi Lang](https://www.taichi-lang.org/)

On Windows, you can install it by running `install.bat` or `pip install taichi` on Linux

Then change value of `ops_backend` from `cupy` to `taichi` in `config.yaml`

## Usage
All VFI nodes are placed in `ComfyUI-Frame-Interpolation/VFI` and require a `IMAGE` containing frames (at least two).

It is recommended to use LoadImages (LoadImagesFromDirectory) from [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/) and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) along side with this extension.

## Example
### Simple workflow
Workflow metadata isn't embeded
Download these two images [anime0.png](./demo_frames/anime0.png) and [anime1.png](./demo_frames/anime0.png) and put them into a folder like `E:\test` in this image.
![](./example.png)

### Complex workflow
It's used in AnimationDiff (can load workflow metadata)
![](All_in_one_v1_3.png)
