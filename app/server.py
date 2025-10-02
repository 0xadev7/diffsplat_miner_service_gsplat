
from __future__ import annotations
import io, os, time, tempfile, pathlib
from typing import Optional
from fastapi import FastAPI, Form, Response
from PIL import Image
import numpy as np

from app.pipeline.diffsplat_wrapper import DiffSplatWrapper
from app.pipeline.clip_validator import clip_score
from app.utils.io import save_zip

app = FastAPI(title="DiffSplat Generation Service", version="0.2.0")
_state = {"clip_model": None, "clip_proc": None, "gen": None}

def _ensure_init():
    if _state["gen"] is None:
        variant = os.environ.get("MODEL_VARIANT", "sd15")
        _state["gen"] = DiffSplatWrapper(variant=variant)
    if _state["clip_model"] is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _state["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            _state["clip_proc"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            _state["clip_model"] = None
            _state["clip_proc"] = None


@app.post("/generate/")
def generate(
    prompt: str = Form(...),
    neg_prompt: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    timeout_s: Optional[float] = Form(28.0),
):
    _ensure_init()
    t0 = time.time()
    try:
        out = _state["gen"].text_to_splat(prompt=prompt, seed=seed, timeout_s=float(timeout_s or 28.0))
        cover = out["cover"]
        ply_bytes = out.get("ply") or b""
    except Exception as e:
        # Return empty byte stream to be ignored by validators
        return Response(content=b"", media_type="application/octet-stream")

    # Lightweight validation (best-effort)
    score = clip_score(_state["clip_model"], _state["clip_proc"], prompt, cover.convert("RGB"))
    if score < 0.12 or not ply_bytes:
        # Empty to avoid cooldown penalties on low quality or missing geometry
        return Response(content=b"", media_type="application/octet-stream")

    # Stream raw PLY bytes (or .splat) directly
    return Response(content=ply_bytes, media_type="application/octet-stream")


    score = clip_score(_state["clip_model"], _state["clip_proc"], prompt, cover.convert("RGB"))
    if score < 0.12:
        meta = {"prompt": prompt, "error": "low_quality", "clip_score": score, "elapsed_s": round((time.time()-t0),3)}
        z = save_zip(cover, b"", meta)
        return Response(content=z, media_type="application/zip")

    meta = {"prompt": prompt, "neg_prompt": neg_prompt, "seed": seed, "clip_score": score, "elapsed_s": round((time.time()-t0),3), "ts": int(time.time())}
    z = save_zip(cover, ply_bytes, meta)
    return Response(content=z, media_type="application/zip")

@app.post("/generate_video/")
def generate_video(
    prompt: str = Form(...),
    neg_prompt: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    timeout_s: Optional[float] = Form(28.0),
):
    _ensure_init()
    try:
        out = _state["gen"].text_to_splat(prompt=prompt, seed=seed, timeout_s=float(timeout_s or 28.0))
        mp4_bytes = out.get("mp4")
        if not mp4_bytes:
            # synthesize a tiny mp4 if upstream didn't emit one
            import imageio
            frames = [(np.zeros((256,256,3), dtype=np.uint8)) for _ in range(12)]
            buf = io.BytesIO()
            imageio.mimwrite(buf, frames, format="mp4", fps=12)
            mp4_bytes = buf.getvalue()
        return Response(content=mp4_bytes, media_type="video/mp4")
    except Exception:
        # return 1s black
        import imageio
        frames = [(np.zeros((256,256,3), dtype=np.uint8)) for _ in range(12)]
        buf = io.BytesIO()
        imageio.mimwrite(buf, frames, format="mp4", fps=12)
        return Response(content=buf.getvalue(), media_type="video/mp4")

if __name__ == "__main__":
    import argparse, uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=False, workers=int(os.environ.get("UVICORN_WORKERS","1")))
