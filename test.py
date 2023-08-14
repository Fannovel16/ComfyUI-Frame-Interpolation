import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import shutil
import torch
import torch.nn.functional as F
import PIL
import torchvision.transforms.functional as transform
from utils import load_file_from_github_release
from models import gmfss_fortuna, ifrnet, ifunet, m2m, rife, sepconv, esai, amt, xvfi, cain, flavr

frame_0 = transform.to_tensor(PIL.Image.open("demo_frames/anime0.png")).unsqueeze(0)
T, C, H, W = frame_0.shape
downscale = int(2 * 8)
frame_0 = F.interpolate(frame_0, size=(8 * (H // downscale), 8 * (W // downscale)))
frame_1 = transform.to_tensor(PIL.Image.open("demo_frames/anime1.png")).unsqueeze(0)
frame_1 = F.interpolate(frame_1, size=(8 * (H // downscale), 8 * (W // downscale)))


if os.path.exists("test_result"):
    shutil.rmtree("test_result")

vfi_node_class = flavr.FLAVR_VFI()
for i, ckpt_name in enumerate(vfi_node_class.INPUT_TYPES()["required"]["ckpt_name"][0]):
    result = vfi_node_class.vfi(ckpt_name, torch.cat([
        frame_0,
        frame_1,
        frame_0,
        frame_1
    ], dim=0).cuda())
    print(f"Generated {result.size(0)} frames")
    frames = [transform.to_pil_image(frame) for frame in result]

    os.makedirs(f"test_result/video{i}", exist_ok=True)
    for j, frame in enumerate(frames):
        frame.save(f"test_result/video{i}/{j}.jpg")
    frames[0].save(f"test_result/video{i}.gif", save_all=True, append_images=frames[1:], optimize=True, duration=1/3, loop=0)
    os.startfile(f"test_result{os.path.sep}video{i}.gif")
#torchvision.io.video.write_video("test.mp4", einops.rearrange(result, "n c h w -> n h w c").cpu(), fps=1)