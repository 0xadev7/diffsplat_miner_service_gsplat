# DiffSplat Competitive Miner — Generation Service (with **gsplat** renderer)

Fast, *30s-or-less* text→3D Gaussian Splat generation service built around
[DiffSplat (ICLR'25)](https://github.com/chenguolin/DiffSplat) with an
**anisotropic 3D Gaussian** renderer powered by **[gsplat]** (CUDA-accelerated).
It keeps **API parity** with the `generation` module in 404‑Repo's
[three-gen-subnet](https://github.com/404-Repo/three-gen-subnet):

- `POST /generate/` → returns a `.zip` with `gaussians.ply`, `preview.png`, `metadata.json`
- `POST /generate_video/` → streams `video/mp4` (orbit video)

**Renderer choice**
- Prefers **gsplat** if available (fast, anisotropic, antialiased).
- Falls back to **Open3D** point rendering if gsplat import fails.

---

## Quick start

```bash
cd diffsplat_miner_service_gsplat
bash ./setup_env.sh

# (Optional) checkpoint warmup
conda activate three-gen-mining
python scripts/download_models.py

# Run service (port 8093)
python -m app.server

# Or using PM2 like three-gen-subnet
pm2 start generation.config.js
```

### Test

```bash
curl -d "prompt=pink bicycle" -X POST http://127.0.0.1:8093/generate_video/ > video.mp4
curl -d "prompt=purple treehouse" -X POST http://127.0.0.1:8093/generate/ --output result.zip
```

---

## gsplat notes

- Install from PyPI (`pip install gsplat`) — it **JIT compiles CUDA on first import**.
  This path is officially supported by the gsplat team. For wheels guidance and
  extra flags see their docs.  
- We feed **Graphdeco‑style PLY** (x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*).
  For preview, we use **DC→RGB** (0.5 + C0·f_dc) and **sigmoid(opacity)**; scales are
  **exp(scale)**; quaternions are taken as‑is (normalized by gsplat).

---

## Self‑validation

Same as before (CLIP+geometry+silhouette). On fail, return **HTTP 204** (empty).