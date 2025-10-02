from __future__ import annotations
import io, json, zipfile
from typing import Dict
from PIL import Image

def save_zip(cover_img: Image.Image, model_bytes: bytes, meta: Dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        imbuf = io.BytesIO()
        cover_img.save(imbuf, format='PNG')
        zf.writestr('cover.png', imbuf.getvalue())
        zf.writestr('model/scene.splat', model_bytes)
        zf.writestr('metadata.json', json.dumps(meta, indent=2))
    return buf.getvalue()
