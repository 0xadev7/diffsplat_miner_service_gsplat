import os, sys, time, numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from loguru import logger

@dataclass
class GenerationOutput:
    ply_bytes: bytes
    preview_rgb: np.ndarray  # HxWx3 in [0,1]
    num_points: int
    timings: Dict[str, float]

def _find_diffsplat_root() -> Optional[str]:
    """
    Locate the DiffSplat repo so that `import src.*` works.
    Priority:
      1) DIFFSPLAT_PATH environment variable
      2) ../DiffSplat relative to project root
      3) ./DiffSplat inside this project
      4) /workspace/DiffSplat (common in containers)
      5) /root/DiffSplat
    """
    from pathlib import Path
    env = os.getenv("DIFFSPLAT_PATH")
    cand = []
    if env:
        cand.append(Path(env))
    here = Path(__file__).resolve()
    proj_root = here.parents[2]
    cand += [
        proj_root.parent / "DiffSplat",
        proj_root / "DiffSplat",
        Path("/workspace/DiffSplat"),
        Path("/root/DiffSplat"),
    ]
    for p in cand:
        if p and p.exists() and (p / "src").exists():
            return str(p)
    return None

class DiffSplatGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_variant = config["generation"]["model_variant"]
        self.device_str = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("FORCE_CUDA")) else "auto"
        self._warm = False
        if config["generation"].get("warmup_on_start", True):
            try:
                self._ensure_imports()
                self._warmup()
            except Exception as e:
                logger.warning(f"Warmup skipped: {e}")

    def _ensure_imports(self):
        root = _find_diffsplat_root()
        if not root:
            raise ImportError(
                "Could not locate DiffSplat repo. "
                "Set DIFFSPLAT_PATH=/path/to/DiffSplat (folder containing 'src') "
                "or clone it next to this project as '../DiffSplat'."
            )
        if root not in sys.path:
            sys.path.insert(0, root)

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
        from .utils import rng_from_seed
        self._ensure_imports()
        r, seed = rng_from_seed(seed)
        t0 = time.time()
        pipe = self._build_text_pipeline(self.model_variant)
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
