import numpy as np
from typing import Literal

def remove_bg(image_rgb: np.ndarray, model: Literal["birefnet","modnet"] = "birefnet") -> np.ndarray:
    if model == "modnet":
        return _modnet(image_rgb)
    return _birefnet(image_rgb)

def _birefnet(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    alpha = np.ones((h,w,1), dtype=np.uint8)*255
    return np.concatenate([image_rgb.astype(np.uint8), alpha], axis=2)

def _modnet(image_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = image_rgb.shape
    alpha = np.ones((h,w,1), dtype=np.uint8)*255
    return np.concatenate([image_rgb.astype(np.uint8), alpha], axis=2)