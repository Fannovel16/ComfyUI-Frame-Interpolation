import itertools
import numpy as np
import vapoursynth as vs
import torch
from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class IFRNet:
    def __init__(self, model, fp16):
        self.fp16 = fp16
        self.cache = False
        self.amount_input_img = 2

        if model == "small":
            from .IFRNet_S_arch import IRFNet_S

            self.model = IRFNet_S()
            model_path = "/workspace/tensorrt/models/IFRNet_S_Vimeo90K.pth"
            check_and_downlaod(model_path)
            self.model.load_state_dict(torch.load(model_path))

        elif model == "large":
            from .IFRNet_L_arch import IRFNet_L

            self.model = IRFNet_L()
            model_path = "/workspace/tensorrt/models/IFRNet_L_Vimeo90K.pth"
            check_and_downlaod(model_path)
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval().cuda()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    def execute(self, I0, I1, timestep):
        if self.fp16:
            I0 = I0.half()
            I1 = I1.half()

        with torch.inference_mode():
            middle = self.model(I0, I1, timestep)

        return middle.detach().squeeze(0).cpu().numpy()
