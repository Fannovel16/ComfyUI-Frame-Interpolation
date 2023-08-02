import yaml
import os
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate it inside repo ComfyUI-Frame-Interpolation by downloading from https://github.com/Fannovel16/ComfyUI-Frame-Interpolation")

