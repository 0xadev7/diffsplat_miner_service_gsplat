from __future__ import annotations
from typing import Optional
from PIL import Image
import numpy as np

def remove_background(img: Image.Image) -> Image.Image:
    try:
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        import torch
        model_id = "briaai/RMBG-1.4"
        processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else None, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        inputs = processor(images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = logits.softmax(dim=1)[:,1,:,:].float().cpu().numpy()[0]
        mask = (probs > 0.5).astype(np.float32)
        rgba = np.array(img.convert("RGBA"), dtype=np.float32)
        rgba[...,3] = (mask * 255.0)
        return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    except Exception:
        return img
