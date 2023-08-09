import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import PIL
import torchvision.transforms.functional as transform
from models import gmfss_fortuna, ifrnet, ifunet, m2m, rife, sepconv, esai

frame_1 = transform.to_tensor(PIL.Image.open("frame1.png")).unsqueeze(0)
frame_3 = transform.to_tensor(PIL.Image.open("frame3.png")).unsqueeze(0)

vfi_node_class = sepconv.SepconvVFI()
for i, ckpt_name in enumerate(vfi_node_class.INPUT_TYPES()["required"]["ckpt_name"][0]):
    result = vfi_node_class.vfi(ckpt_name, torch.cat([
        frame_1,
        frame_3,
        frame_1
    ], dim=0).cuda(), batch_size = 2)
    print(f"Generated {result.size(0)} frames")
    frames = [transform.to_pil_image(frame) for frame in result]
    frames[0].save(f"test{i}.gif", save_all=True, append_images=frames[1:], optimize=True, duration=1/5, loop=0)
    os.system(f"test{i}.gif")
#torchvision.io.video.write_video("test.mp4", einops.rearrange(result, "n c h w -> n h w c").cpu(), fps=1)