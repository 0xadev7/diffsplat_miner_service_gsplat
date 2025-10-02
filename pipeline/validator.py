import numpy as np
from typing import Tuple, Dict
from loguru import logger

def _clip_score(prompt: str, preview_rgb: np.ndarray) -> float:
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transform("ViT-L-14", pretrained="openai")
    model.eval()
    with torch.no_grad():
        img = preprocess(_pil_from_nd(preview_rgb)).unsqueeze(0)
        txt = open_clip.tokenize([prompt])
        if torch.cuda.is_available():
            model = model.cuda(); img = img.cuda(); txt = txt.cuda()
        feats_i = model.encode_image(img)
        feats_t = model.encode_text(txt)
        feats_i = feats_i / feats_i.norm(dim=-1, keepdim=True)
        feats_t = feats_t / feats_t.norm(dim=-1, keepdim=True)
        sim = (feats_i @ feats_t.T).item()
        return float(sim)

def _pil_from_nd(img: np.ndarray):
    from PIL import Image
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype("uint8")
    return Image.fromarray(img)

def _silhouette_coverage(img: np.ndarray) -> float:
    import cv2, numpy as np
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = (mask > 0).astype(np.float32).mean()
    return float(fg)

async def quick_validate(prompt: str, preview_rgb: np.ndarray, num_points: int) -> Tuple[bool, Dict]:
    from app.pipeline_config import CONFIG
    cfgv = CONFIG["validation"]
    min_pts = int(cfgv["min_points"]); max_pts = int(cfgv["max_points"])
    if num_points < min_pts or num_points > max_pts:
        return False, {"reason": "point_count", "num_points": int(num_points)}

    try:
        sim = _clip_score(prompt, preview_rgb)
    except Exception as e:
        logger.warning(f"CLIP scoring failed: {e}")
        sim = 1.0

    if sim < float(cfgv["min_clip_sim"]):
        return False, {"reason": "clip", "score": sim}

    cov = _silhouette_coverage(preview_rgb)
    if not (float(cfgv["min_silhouette"]) <= cov <= float(cfgv["max_silhouette"])):
        return False, {"reason": "silhouette", "coverage": cov}

    return True, {"clip": sim, "silhouette": cov}