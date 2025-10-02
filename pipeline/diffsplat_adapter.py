import os, time, numpy as np, io, json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from PIL import Image
from loguru import logger

from .utils import rng_from_seed

@dataclass
class GenerationOutput:
    ply_bytes: bytes
    preview_rgb: np.ndarray  # HxWx3 in [0,1]
    num_points: int
    timings: Dict[str, float]

class DiffSplatGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_variant = config["generation"]["model_variant"]
        self.device_str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.environ.get("FORCE_CUDA","") else "auto"
        self._warm = False
        if config["generation"].get("warmup_on_start", True):
            try:
                self._ensure_imports()
                self._warmup()
            except Exception as e:
                logger.warning(f"Warmup skipped: {e}")

    def _ensure_imports(self):
        import sys, pathlib
        root = pathlib.Path(__file__).resolve().parents[2]
        diffsplat = root.parent / "DiffSplat"
        if diffsplat.exists():
            sys.path.insert(0, str(diffsplat))
        from src.pipelines import build_text_pipeline  # type: ignore
        from src.utils.ply_utils import write_ply_bytes  # type: ignore
        from src.utils.camera_utils import get_default_camera  # type: ignore
        self._build_text_pipeline = build_text_pipeline
        self._write_ply_bytes = write_ply_bytes
        self._get_default_camera = get_default_camera

    def _warmup(self):
        if self._warm: return
        self._ensure_imports()
        _ = self._build_text_pipeline(self.model_variant)
        self._warm = True

    async def generate(self, prompt: str, seed: Optional[int] = None) -> GenerationOutput:
        self._ensure_imports()
        r, seed = rng_from_seed(seed)
        t0 = time.time()

        pipe = self._build_text_pipeline(self.model_variant)

        # NOTE: Replace with the correct API as needed
        result = pipe.generate(prompt=prompt, seed=seed, steps=8)
        t2 = time.time()

        ply_bytes = self._write_ply_bytes(result)
        num_points = int(result.get("num_points", 0) or 0)
        preview_rgb = result.get("preview", None)
        if preview_rgb is None:
            cam = self._get_default_camera()
            preview_rgb = pipe.render_preview(result, cam)

        t3 = time.time()
        timings = {"generate": t2 - t0, "export": t3 - t2, "total": t3 - t0}
        return GenerationOutput(ply_bytes=ply_bytes, preview_rgb=preview_rgb, num_points=num_points, timings=timings)