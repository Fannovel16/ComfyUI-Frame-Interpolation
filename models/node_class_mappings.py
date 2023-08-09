#from .esai import EISAI_VFI
from .gmfss_fortuna import GMFSS_Fortuna_VFI
from .ifrnet import IFRNet_VFI
from .ifunet import IFUnet_VFI
from .m2m import M2M_VFI
from .rife import RIFE_VFI
from .sepconv import SepconvVFI
from .amt import AMT_VFI

NODE_CLASS_MAPPINGS = {
    #"ESAI VFI": EISAI_VFI,
    "GMFSS Fortuna VFI": GMFSS_Fortuna_VFI,
    "IFRNet VFI": IFRNet_VFI,
    "IFUnet VFI": IFUnet_VFI,
    "M2M VFI": M2M_VFI,
    "RIFE VFI": RIFE_VFI,
    "Sepconv VFI": SepconvVFI,
    "AMT VFI": AMT_VFI
}