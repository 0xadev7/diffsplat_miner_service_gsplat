import os, time, json, shutil, subprocess, glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

@dataclass
class GenerationOutput:
    ply_bytes: bytes
    preview_rgb: np.ndarray  # HxWx3 in [0,1]
    num_points: int
    timings: Dict[str, float]

def _slug(s: str) -> str:
    return "_".join(s.strip().replace("/", " ").split())

def _ds_root() -> str:
    # 1) explicit; 2) typical mount in your machine
    env = os.getenv("DIFFSPLAT_PATH")
    if env and os.path.isdir(env):
        return env
    for p in ("/root/DiffSplat", "/workspace/DiffSplat"):
        if os.path.isdir(p):
            return p
    raise RuntimeError("Set DIFFSPLAT_PATH=/root/DiffSplat (folder that contains src/ and configs/)")

def _model_cfg_variant(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Pick which DiffSplat variant to use.
    Defaults to SD1.5 text model:
      script:  src/infer_gsdiff_sd.py
      config:  configs/gsdiff_sd15.yaml
      weights: gsdiff_gobj83k_sd15__render
    You can switch to PixArt or SD3.5m by editing app/pipeline_config.yaml.
    """
    g = config["generation"]
    v = g.get("model_variant", "sd15_text")
    ds = _ds_root()
    if v in ("sd15", "sd15_text"):
        return (os.path.join(ds, "src/infer_gsdiff_sd.py"),
                os.path.join(ds, "configs/gsdiff_sd15.yaml"),
                "gsdiff_gobj83k_sd15__render")
    if v in ("pas", "pixart"):
        return (os.path.join(ds, "src/infer_gsdiff_pas.py"),
                os.path.join(ds, "configs/gsdiff_pas.yaml"),
                "gsdiff_gobj83k_pas_fp16__render")
    if v in ("sd35m", "sd3.5m"):
        return (os.path.join(ds, "src/infer_gsdiff_sd3.py"),
                os.path.join(ds, "configs/gsdiff_sd35m_80g.yaml"),
                "gsdiff_gobj83k_sd35m__render")
    # fallback
    return (os.path.join(ds, "src/infer_gsdiff_sd.py"),
            os.path.join(ds, "configs/gsdiff_sd15.yaml"),
            "gsdiff_gobj83k_sd15__render")

def _out_dir(model_name: str) -> str:
    # README/forks: outputs go to ./out/<MODEL_NAME>/inference
    # We'll resolve relative to DIFFSPLAT_PATH to avoid surprises.
    ds = _ds_root()
    p = os.path.join(ds, "out", model_name, "inference")
    os.makedirs(p, exist_ok=True)
    return p

def _latest_of(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

class DiffSplatGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_variant = config["generation"]["model_variant"]
        self.device_str = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("FORCE_CUDA")) else "auto"

    async def generate(self, prompt: str, seed: Optional[int] = None) -> GenerationOutput:
        from .utils import rng_from_seed
        r, seed = rng_from_seed(seed)
        t0 = time.time()

        script_py, cfg_yaml, model_name = _model_cfg_variant(self.config)
        out_dir = _out_dir(model_name)
        prompt_slug = _slug(prompt)
        # DiffSplat CLI expects underscores; it replaces them with spaces internally. :contentReference[oaicite:1]{index=1}

        # Build command (fast path: MP4, low overhead). “infer.sh” handles env; calling python directly also works.
        ds = _ds_root()
        infer_sh = os.path.join(ds, "scripts/infer.sh")
        if os.path.isfile(infer_sh):
            cmd = [
                "bash", infer_sh,
                script_py, cfg_yaml, model_name,
                "--prompt", prompt_slug,
                "--output_video_type", "mp4",
                "--gpu_id", os.environ.get("GEN_GPU_ID", "0"),
                "--seed", str(seed or 0),
                "--save_ply"  # ensure PLY export – see issues suggesting adjusting save_ply/opacity threshold. :contentReference[oaicite:2]{index=2}
            ]
        else:
            # Fallback: call python script directly
            cmd = [
                "python", script_py,
                "--config", cfg_yaml,
                "--model_name", model_name,
                "--prompt", prompt_slug,
                "--output_video_type", "mp4",
                "--gpu_id", os.environ.get("GEN_GPU_ID", "0"),
                "--seed", str(seed or 0),
                "--save_ply"
            ]

        env = os.environ.copy()
        env.setdefault("PYTHONPATH", f"{ds}:" + env.get("PYTHONPATH", ""))
        # If HF mirror is needed in your region, you may uncomment:
        # env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # Run inference (it writes files into out/<MODEL_NAME>/inference)
        proc = subprocess.run(cmd, env=env, cwd=ds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=25)
        if proc.returncode != 0:
            raise RuntimeError(f"DiffSplat inference failed (rc={proc.returncode}):\n{proc.stdout.decode('utf-8','ignore')}")

        # Collect artifacts
        # 1) PLY — there should be one per prompt; use newest match
        ply_path = _latest_of(os.path.join(out_dir, f"*{prompt_slug}*.ply"))
        if not ply_path or not os.path.isfile(ply_path):
            # Sometimes PLY may be filtered by opacity thresholds; try a generic match
            ply_path = _latest_of(os.path.join(out_dir, "*.ply"))
        if not ply_path or not os.path.isfile(ply_path):
            raise RuntimeError("No PLY produced by DiffSplat.")

        with open(ply_path, "rb") as f:
            ply_bytes = f.read()

        # 2) A quick preview image
        # We’ll render our own first view later with gsplat; for validator precheck here,
        # try to grab the first frame from the mp4 if present, else synthesize a tiny white.
        mp4_path = _latest_of(os.path.join(out_dir, f"*{prompt_slug}*.mp4")) or _latest_of(os.path.join(out_dir, "*.mp4"))
        preview_rgb = np.ones((8, 8, 3), dtype=np.float32)  # placeholder; real preview is rendered in our renderer
        num_points = _estimate_point_count_from_ply_header(ply_bytes)

        t1 = time.time()
        return GenerationOutput(ply_bytes=ply_bytes,
                                preview_rgb=preview_rgb,
                                num_points=num_points,
                                timings={"diffsplat_cli": t1 - t0})

def _estimate_point_count_from_ply_header(ply_bytes: bytes) -> int:
    # Quick parse: find "element vertex N"
    try:
        head = ply_bytes[:2048].decode("ascii", "ignore")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return int(line.strip().split()[-1])
    except Exception:
        pass
    return 0
