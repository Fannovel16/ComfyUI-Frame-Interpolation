import os
from pathlib import Path
import sys
import platform

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

s_param = '-s' if "python_embeded" in sys.executable else '' 

def temp_add_cuda_home():
    if "CUDA_HOME" not in os.environ:
        if platform.system() == "Windows":
            os.environ["CUDA_HOME"] = str((Path(__file__).parent / "models/ops/cupy_ops/cuda_dlls").resolve())
        else:
            import torch
            torch_lib_path = Path(torch.__file__).parent / "lib"
            torch_lib_path = str(torch_lib_path.resolve())
            if os.path.exists(torch_lib_path):
                os.environ["CUDA_HOME"] = torch_lib_path
            else:
                os.environ["CUDA_HOME"] = "/usr/local/cuda/"

def install_cupy():
    try:
        temp_add_cuda_home()
        print(os.environ["CUDA_HOME"])
        import cupy
        print("CuPy is already installed.")
    except Exception:
        print("Uninstall cupy if existed...")
        os.system(f'"{sys.executable}" {s_param} -m pip uninstall -y cupy-wheel cupy-cuda102 cupy-cuda110 cupy-cuda111 cupy-cuda11x cupy-cuda12x')
        print("Installing cupy...")
        cuda_ver = None

        #./models/ops/cupy_ops/cuda_dlls
        if platform.system() == "Windows":
            cuda_ver = "12x"

        if (cuda_ver is None) and ("CUDA_HOME" not in os.environ):
            import torch
            torch_lib_path = Path(torch.__file__).parent / "lib"
            torch_lib_path = str(torch_lib_path.resolve())
            if os.path.exists(torch_lib_path):
                nvrtc = filter(lambda lib_file: "nvrtc-builtins" in lib_file, os.listdir(torch_lib_path))
                nvrtc = list(nvrtc)[0]
                cuda_ver = get_cuda_ver(nvrtc)
        
        cupy_package = f"cupy-cuda{cuda_ver}" if cuda_ver is not None else "cupy-wheel"
        os.system(f'"{sys.executable}" {s_param} -m pip install {cupy_package}')

with open("requirements-no-cupy.txt", 'r') as f:
    for package in f.readlines():
        package = package.strip()
        print(f"Installing {package}...")
        os.system(f'"{sys.executable}" {s_param} -m pip install {package}')

print("Checking cupy...")
install_cupy()
