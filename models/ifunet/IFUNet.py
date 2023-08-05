import itertools
import numpy as np
import vapoursynth as vs
from .IFUNet_arch import IFUNetModel
import torch
from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class IFUNet:
    def __init__(self):
        self.amount_input_img = 2
        self.cache = False

        torch.set_grad_enabled(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        model_path = "/workspace/tensorrt/models/IFUNet.pth"
        check_and_download(model_path)

        self.model = IFUNetModel()
        self.model.load_state_dict(torch.load(model_path), False)
        self.model.cuda().eval()

    def execute(self, I0, I1, timestep):
        with torch.inference_mode():
            middle = self.model(I0, I1, timestep).cpu()
        return middle
