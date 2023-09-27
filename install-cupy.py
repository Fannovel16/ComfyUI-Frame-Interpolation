import os
from pathlib import Path
import sys

def get_cuda_ver(nvrtc):
    if ('102' in nvrtc) or ('10.2' in nvrtc):
        return '102'
    if '110' in nvrtc or ('11.0' in nvrtc):
        return '110'
    if '111' in nvrtc or ('11.1' in nvrtc):
        return '111'
    if '11' in nvrtc:
        return '11x'
    if '12' in nvrtc:
        return '12x'
    return None

try:
    import cupy
    print("CuPy is already installed.")
except:
    cuda_ver = None
    cuda_ver = None
    if "CUDA_HOME" not in os.environ:
        import torch
        torch_lib_path = Path(torch.__file__).parent / "lib"
        torch_lib_path = str(torch_lib_path.resolve())
        if os.path.exists(torch_lib_path):
            nvrtc = filter(lambda lib_file: "nvrtc-builtins" in lib_file, os.listdir(torch_lib_path))
            nvrtc = list(nvrtc)[0]
            cuda_ver = get_cuda_ver(nvrtc)

    s_param = '-s' if "python_embeded" in sys.executable else '' 
    cupy_package = f"cupy-cuda{cuda_ver}" if cuda_ver is not None else "cupy-wheel"
    os.system(f'"{sys.executable}" {s_param} -m pip install {cupy_package}')