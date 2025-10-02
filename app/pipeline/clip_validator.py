from __future__ import annotations
import torch
from PIL import Image

@torch.no_grad()
def clip_score(clip_model, clip_proc, prompt: str, img: Image.Image) -> float:
    if clip_model is None or clip_proc is None:
        return 1.0
    inputs = clip_proc(text=[prompt], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(clip_model.device) for k,v in inputs.items()}
    out = clip_model(**inputs)
    sim = torch.nn.functional.cosine_similarity(out.text_embeds, out.image_embeds).mean().item()
    return float(sim)
