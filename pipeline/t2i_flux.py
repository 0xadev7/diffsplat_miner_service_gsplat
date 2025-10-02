from typing import Optional
import torch
from diffusers import FluxPipeline

def sample_flux_schnell(prompt: str, seed: Optional[int] = None, steps: int = 2, height=512, width=512):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0, height=height, width=width, generator=g).images[0]
    return img