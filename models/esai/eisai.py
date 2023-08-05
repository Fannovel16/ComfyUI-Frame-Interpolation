import itertools
import numpy as np
import vapoursynth as vs
from .eisai_arch import SoftsplatLite, DTM, RAFT, interpolate
import torch
from .download import check_and_download


# https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
class EISAI:
    def __init__(self):
        self.amount_input_img = 2
        self.cache = False

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # load models
        device = torch.device("cuda")

        ssl = SoftsplatLite()
        dtm = DTM()

        ssl_path = "/workspace/tensorrt/models/eisai_ssl.pt"
        dtm_path = "/workspace/tensorrt/models/eisai_dtm.pt"
        raft_path = "/workspace/tensorrt/models/eisai_anime_interp_full.ckpt"

        check_and_download(ssl_path)
        check_and_download(dtm_path)
        check_and_download(raft_path)

        ssl.load_state_dict(torch.load(ssl_path))
        dtm.load_state_dict(torch.load(dtm_path))
        self.raft = RAFT(path=raft_path).eval().to(device)
        self.ssl = ssl.to(device).eval()
        self.dtm = dtm.to(device).eval()

    def execute(self, I0, I1, timestep):
        with torch.inference_mode():
            middle = interpolate(self.raft, self.ssl, self.dtm, I0, I1, t=timestep)
            middle = middle.detach().cpu().numpy()
        return middle
