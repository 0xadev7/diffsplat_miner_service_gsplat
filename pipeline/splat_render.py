import io, math, numpy as np, imageio.v2 as imageio
from typing import Dict, Iterable
from PIL import Image

_USE_GSPLAT = False
try:
    import gsplat  # noqa: F401
    from .gsplat_render import render_frames_gsplat
    _USE_GSPLAT = True
except Exception:
    _USE_GSPLAT = False

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

def render_preview_png(ply_bytes: bytes, cfg: Dict) -> bytes:
    if _USE_GSPLAT:
        frames = render_frames_gsplat(ply_bytes, cfg, return_first_only=True)
        img = next(frames)
    else:
        frames = _render_frames_o3d(ply_bytes, cfg, take_first_only=True)
        img = next(frames)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()

def render_orbit_mp4(ply_bytes: bytes, cfg: Dict) -> Iterable[bytes]:
    writer = imageio.get_writer("<bytes>", format="ffmpeg", mode="I",
                                fps=cfg["fps"], codec="h264", bitrate="2M")
    try:
        if _USE_GSPLAT:
            frames = render_frames_gsplat(ply_bytes, cfg, return_first_only=False)
        else:
            frames = _render_frames_o3d(ply_bytes, cfg, take_first_only=False)
        for img in frames:
            writer.append_data(img)
        writer.close()
        yield writer.getvalue()
    finally:
        try:
            writer.close()
        except Exception:
            pass

def _render_frames_o3d(ply_bytes: bytes, cfg: Dict, take_first_only=False):
    import open3d as o3d
    import numpy as np
    w, h = cfg["width"], cfg["height"]
    ply_io = io.BytesIO(ply_bytes)
    with open("/tmp/_tmp_splat.ply", "wb") as f:
        f.write(ply_io.read())
    pcd = o3d.io.read_point_cloud("/tmp/_tmp_splat.ply")

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=16))

    vis = o3d.visualization.rendering.OffscreenRenderer(w, h)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    vis.scene.set_sun_light([1,1,1], 6500, 1.0)
    vis.scene.add_point_cloud("pcd", pcd, mat)

    center = np.array(cfg.get("lookat", [0,0,0]), dtype=float)
    up = np.array(cfg.get("up", [0,1,0]), dtype=float)
    fov = float(cfg.get("fov_deg", 45.0))

    for i, (x,y,z) in enumerate(_orbit_poses(cfg)):
        eye = np.array([x,y,z], dtype=float)
        vis.setup_camera(fov, center, eye, up)
        img = vis.render_to_image()
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.repeat(img[...,None], 3, axis=2)
        if img.shape[2] == 4:
            img = img[..., :3]
        yield img
        if take_first_only:
            break

    vis.scene.clear_geometry()
    vis.release_window()