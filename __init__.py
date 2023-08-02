import torch

import os
import sys
import latent_preview
import comfy

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    device = comfy.model_management.get_torch_device()
    latent_image = latent["samples"]

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    previewer = latent_preview.get_previewer(device, model.model.latent_format)

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

class Gradually_More_Denoise_KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    
                    "start_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "denoise_increment": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1}),
                    "denoise_increment_steps": ("INT", {"default": 20, "min": 1, "max": 10000})
                     },
                "optional": { "optional_vae": ("VAE",) }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Frame Interpolation"

    def sample(self, model, positive, negative, latent_image, vae, 
               seed, steps, cfg, sampler_name, scheduler,start_denoise, denoise_increment, denoise_steps):
        if start_denoise + denoise_increment * denoise_steps > 1.0:
            raise Exception(f"Max denoise strength can't over 1.0 (start_denoise={start_denoise}, denoise_increment={denoise_increment}, denoise_steps={denoise_steps}")

        outs = []
        for latent in latent_image:
            out = latent.copy()
            gradually_denoising_latents = [
                common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=start_denoise + denoise_increment * i)
                for i in range(denoise_steps)
            ]
            samples = torch.cat(list(map(lambda x: x["samples"], gradually_denoising_latents)), dim=0)
            out["samples"] = samples
        
        outs = torch.cat(outs, dim=0)
        return (model, positive, negative, outs, vae)

NODE_CLASS_MAPPINGS = {
    "KSampler Gradually Adding More Denoise (efficient)": Gradually_More_Denoise_KSampler
}