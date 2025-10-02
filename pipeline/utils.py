import io, os, random, numpy as np
from typing import Optional

def rng_from_seed(seed: Optional[int]):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "little")
    r = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))
    return r, seed

def to_uint8(img_f32):
    img = np.clip(img_f32 * 255.0, 0, 255).astype("uint8")
    return img