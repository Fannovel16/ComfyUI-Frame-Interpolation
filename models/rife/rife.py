import itertools
import numpy as np
import vapoursynth as vs
import torch
from .rife_arch import IFNet
from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class RIFE:
    def __init__(self, scale, fastmode, ensemble, model_version, fp16):
        self.scale = scale
        self.fastmode = fastmode
        self.ensemble = ensemble
        self.model_version = model_version
        self.fp16 = fp16
        self.cache = False
        self.amount_input_img = 2

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if model_version == "rife40":
            model_path = "/workspace/tensorrt/models/rife40.pth"
            arch_ver = "4.0"
        elif model_version == "rife41":
            model_path = "/workspace/tensorrt/models/rife41.pth"
            arch_ver = "4.0"
        elif model_version == "rife42":
            model_path = "/workspace/tensorrt/models/rife42.pth"
            arch_ver = "4.2"
        elif model_version == "rife43":
            model_path = "/workspace/tensorrt/models/rife43.pth"
            arch_ver = "4.3"
        elif model_version == "rife44":
            model_path = "/workspace/tensorrt/models/rife44.pth"
            arch_ver = "4.3"
        elif model_version == "rife45":
            model_path = "/workspace/tensorrt/models/rife45.pth"
            arch_ver = "4.5"
        elif model_version == "rife46":
            model_path = "/workspace/tensorrt/models/rife46.pth"
            arch_ver = "4.6"
        elif model_version == "sudo_rife4":
            model_path = (
                "/workspace/tensorrt/models/sudo_rife4_269.662_testV1_scale1.pth"
            )
            arch_ver = "4.0"

        check_and_download(model_path)
        self.model = IFNet(arch_ver=arch_ver)
        self.model.load_state_dict(torch.load(model_path), False)

        self.model.eval().cuda()

        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            self.model.half()

    def execute(self, I0, I1, timestep):
        scale_list = [8 / self.scale, 4 / self.scale, 2 / self.scale, 1 / self.scale]

        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(
                I0,
                I1,
                scale_list=scale_list,
                fastmode=self.fastmode,
                ensemble=self.ensemble,
                timestep=timestep,
            )

        middle = middle.detach().squeeze(0).cpu().numpy()
        return middle
