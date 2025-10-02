import os, io, asyncio, time, json, zipfile, traceback
from typing import Optional

from fastapi import FastAPI, Form, Body, Response
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from .api_schemas import GenerateRequest, GenerateVideoRequest, HealthResponse
from .pipeline_config import CONFIG
from pipeline.diffsplat_adapter import DiffSplatGenerator, GenerationOutput
from pipeline.splat_render import render_preview_png, render_orbit_mp4
from pipeline.validator import quick_validate

app = FastAPI(title="DiffSplat Competitive Miner - Generation Service (gsplat)")
GEN = DiffSplatGenerator(config=CONFIG)

def form_or_json_prompt(prompt: Optional[str], seed: Optional[int], body: Optional[dict]):
    if prompt is None and body is not None:
        prompt = body.get("prompt")
        seed = body.get("seed", seed)
    if not prompt or not isinstance(prompt, str):
        return None, seed
    return prompt.strip(), seed

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=GEN.device_str,
        model_variant=GEN.model_variant,
    )

@app.post("/generate/")
@app.post("/generate")
async def generate(
    prompt: Optional[str] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    body: Optional[GenerateRequest] = Body(default=None),
):
    req_json = body.dict() if body else None
    prompt, seed = form_or_json_prompt(prompt, seed, req_json)
    if prompt is None:
        return JSONResponse({"error": "Missing 'prompt'."}, status_code=400)

    logger.info(f"/generate :: prompt='{prompt[:120]}' seed={seed}")
    t0 = time.time()
    try:
        g: GenerationOutput = await GEN.generate(prompt, seed=seed)
        ok, val_meta = await quick_validate(prompt, g.preview_rgb, g.num_points)
        if not ok:
            logger.warning(f"Validation failed: {val_meta}")
            return Response(status_code=204)

        meta = dict(
            prompt=prompt,
            seed=seed,
            model_variant=GEN.model_variant,
            num_points=g.num_points,
            timings=g.timings,
            validation=val_meta,
            camera=CONFIG["video"],
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("gaussians.ply", g.ply_bytes)
            zf.writestr("preview.png", render_preview_png(g.ply_bytes, CONFIG["video"]))
            zf.writestr("metadata.json", json.dumps(meta, indent=2))
        buf.seek(0)
        elapsed = time.time() - t0
        logger.info(f"/generate OK in {elapsed:.2f}s, points={g.num_points}")
        headers = {"Content-Disposition": 'attachment; filename="result.zip"'}
        return Response(content=buf.read(), media_type="application/zip", headers=headers)
    except Exception as e:
        logger.error(f"/generate error: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": "internal_error"}, status_code=500)

@app.post("/generate_video/")
@app.post("/generate_video")
async def generate_video(
    prompt: Optional[str] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    body: Optional[GenerateVideoRequest] = Body(default=None),
):
    req_json = body.dict() if body else None
    prompt, seed = form_or_json_prompt(prompt, seed, req_json)
    if prompt is None:
        return JSONResponse({"error": "Missing 'prompt'."}, status_code=400)

    logger.info(f"/generate_video :: prompt='{prompt[:120]}' seed={seed}")
    try:
        g: GenerationOutput = await GEN.generate(prompt, seed=seed)
        ok, val_meta = await quick_validate(prompt, g.preview_rgb, g.num_points)
        if not ok:
            logger.warning(f"Validation failed: {val_meta}")
            return Response(status_code=204)

        def iter_video():
            for chunk in render_orbit_mp4(g.ply_bytes, CONFIG["video"]):
                yield chunk
        headers = {"Content-Disposition": 'attachment; filename="video.mp4"'}
        return StreamingResponse(iter_video(), media_type="video/mp4", headers=headers)
    except Exception as e:
        logger.error(f"/generate_video error: {e}\n{traceback.format_exc()}")
        return JSONResponse({"error": "internal_error"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8093"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", workers=1)