import io, math, numpy as np
from typing import Dict

def _orbit_poses(cfg: Dict):
    secs = cfg["seconds"]
    fps = cfg["fps"]
    n = int(secs * fps)
    for i in range(n):
        t = i / n
        az = math.radians(t * cfg["orbit_degrees"])
        el = math.radians(cfg["elevation_deg"])
        r = cfg["cam_radius"]
        x = r * math.sin(az) * math.cos(el)
        y = r * math.sin(el)
        z = r * math.cos(az) * math.cos(el)
        yield (x, y, z)

def _build_lookat(world_eye, world_center, world_up) -> np.ndarray:
    eye = np.array(world_eye, dtype=np.float32)
    center = np.array(world_center, dtype=np.float32)
    up = np.array(world_up, dtype=np.float32)
    f = center - eye; f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u); s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s; M[1, :3] = u; M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M

def _intrinsics_from_fov(width, height, fov_deg):
    f = 0.5 * height / math.tan(math.radians(fov_deg) * 0.5)  # vertical FOV
    fx = f; fy = f
    cx = (width - 1) / 2.0; cy = (height - 1) / 2.0
    K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float32)
    return K

def _parse_ply_bytes(ply_bytes: bytes):
    from plyfile import PlyData
    import io, numpy as np
    f = io.BytesIO(ply_bytes)
    ply = PlyData.read(f)
    v = ply["vertex"].data
    def col(names):
        return np.stack([v[n] for n in names], axis=-1).astype(np.float32)
    means = col(["x","y","z"])
    f_dc = col(["f_dc_0","f_dc_1","f_dc_2"])
    opacity_raw = v["opacity"].astype(np.float32)
    scales_raw = col(["scale_0","scale_1","scale_2"])
    quats = col(["rot_0","rot_1","rot_2","rot_3"])

    C0 = 0.28209479177387814
    colors = np.clip(0.5 + C0 * f_dc, 0.0, 1.0).astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_raw))
    scales = np.exp(scales_raw)
    return means, quats, scales, opacities, colors

def render_frames_gsplat(ply_bytes: bytes, cfg: Dict, return_first_only=False):
    import torch, gsplat
    w, h = int(cfg["width"]), int(cfg["height"])
    lookat = cfg.get("lookat", [0,0,0]); up = cfg.get("up", [0,1,0])
    fov = float(cfg.get("fov_deg", 45.0))

    means_np, quats_np, scales_np, op_np, colors_np = _parse_ply_bytes(ply_bytes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    means = torch.from_numpy(means_np).to(device)
    quats = torch.from_numpy(quats_np).to(device)
    scales = torch.from_numpy(scales_np).to(device)
    opacities = torch.from_numpy(op_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)

    background = torch.ones(1, 3, device=device)
    Ks = torch.from_numpy(_intrinsics_from_fov(w, h, fov)).to(device).unsqueeze(0)

    for (x,y,z) in _orbit_poses(cfg):
        view = _build_lookat([x,y,z], lookat, up)
        view = torch.from_numpy(view).to(device).unsqueeze(0)  # [1,4,4]

        rgb, alpha, meta = gsplat.rendering.rasterization(
            means=means[None, ...],
            quats=quats[None, ...],
            scales=scales[None, ...],
            opacities=opacities[None, ...],
            colors=colors[None, ...],
            viewmats=view[None, ...],   # [1,1,4,4]
            Ks=Ks[None, ...],           # [1,1,3,3]
            width=w, height=h,
            sh_degree=None,
            rasterize_mode="antialiased",
            backgrounds=background[None, ...],
            packed=True,
            tile_size=16,
        )
        if rgb.ndim == 5 and rgb.shape[-1] == 3:
            img = rgb[0,0].detach().clamp(0,1).mul(255).byte().cpu().numpy()
        elif rgb.ndim == 5 and rgb.shape[2] == 3:
            img = rgb[0,0].permute(1,2,0).detach().clamp(0,1).mul(255).byte().cpu().numpy()
        else:
            arr = rgb.detach().cpu()
            if arr.shape[-1] == 3:
                img = (arr[0,0].clamp(0,1).numpy() * 255).astype("uint8")
            else:
                img = (arr[0,0].permute(1,2,0).clamp(0,1).numpy() * 255).astype("uint8")
        yield img
        if return_first_only:
            break