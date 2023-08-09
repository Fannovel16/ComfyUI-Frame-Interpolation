import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import PIL
import torchvision.transforms.functional as transform
from models import gmfss_fortuna, ifrnet, ifunet, m2m, rife, sepconv, esai, amt

frame_0 = transform.to_tensor(PIL.Image.open("demo_frames/real0.png")).unsqueeze(0)
frame_0 = F.interpolate(frame_0, scale_factor=0.5)
frame_1 = transform.to_tensor(PIL.Image.open("demo_frames/real1.png")).unsqueeze(0)
frame_1 = F.interpolate(frame_1, scale_factor=0.5)


vfi_node_class = amt.AMT_VFI()
#for i, ckpt_name in enumerate(vfi_node_class.INPUT_TYPES()["required"]["ckpt_name"][0]):
for i, ckpt_name in enumerate(["gopro_amt-s.pth"]):
    result = vfi_node_class.vfi(ckpt_name, torch.cat([
        frame_0,
        frame_1,
        frame_0
    ], dim=0).cuda(), batch_size = 2, multipler=15)
    print(f"Generated {result.size(0)} frames")
    frames = [transform.to_pil_image(frame) for frame in result]
    frames[0].save(f"test{i}.gif", save_all=True, append_images=frames[1:], optimize=True, duration=1/5, loop=0)
    os.system(f"test{i}.gif")
#torchvision.io.video.write_video("test.mp4", einops.rearrange(result, "n c h w -> n h w c").cpu(), fps=1)