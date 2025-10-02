# DiffSplat Competitive Generation Service

Production-ready generation microservice built around **DiffSplat** with endpoints compatible with
the *Three-Gen Subnet* miner's generation API.

- Endpoints: `/generate/` and `/generate_video/`
- Form-data interface: `prompt=...`
- Self-validation: lightweight CLIP similarity
- Background removal: optional BRIA RMBG v1.4 (commercial friendly); falls back gracefully
- CUDA 12.4-ready Dockerfile including fixes for `glm/glm.hpp` and `uintptr_t` compile errors
- Tuned for speed; designed to return within ~30s when models are preloaded

Port defaults to **8093** (as in the reference quick test).

## Docker Quickstart (CUDA 12.4)

```bash
docker build -t diffsplat-gen:cuda124 .
docker run --gpus all -p 8093:8093 -v $PWD/cache:/cache diffsplat-gen:cuda124
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
curl -d "prompt=wooden chair" -X POST http://127.0.0.1:8093/generate/ --output result.zip
```


## No-Docker Install on RunPod

```bash
bash scripts/install_runpod.sh
python -m app.server --host 0.0.0.0 --port 8093
```

## Using the Async Miner (fan-out + blacklist)

Provide a small JSON listing validator pull/push endpoints and any blacklisted UIDs, then:

```python
from app.pipeline.async_miner import AsyncMiner

validators = [
  {"uid": 101, "pull_url": "http://validator-101/pull", "push_url": "http://validator-101/push"},
  {"uid": 180, "pull_url": "http://validator-180/pull", "push_url": "http://validator-180/push"},
]
miner = AsyncMiner(validators=validators, service_url="http://127.0.0.1:8093", blacklist=[180], workers=4)
miner.run_forever()
```


### API Compatibility Update
- `/generate/` now returns **raw PLY bytes** (`application/octet-stream`) exactly like the base miner.
- `/generate_video/` returns `video/mp4`.

Example:
```bash
curl -s -d "prompt=wooden chair" -X POST http://127.0.0.1:8093/generate/ > scene.ply
```
