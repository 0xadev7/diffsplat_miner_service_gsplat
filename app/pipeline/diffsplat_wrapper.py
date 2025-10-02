from __future__ import annotations
import os, time, tempfile, subprocess, shutil, glob, io
from typing import Dict
from PIL import Image

from app.utils.logging import get_logger, time_block


class DiffSplatWrapper:
    logger = get_logger("diffsplat")

    def __init__(self, repo_dir: str = "DiffSplat", variant: str = "sd15"):
        self.repo_dir = repo_dir
        self.variant = variant  # sd15 | sd35m | pas (pixart-sigma)

    def _infer_entry(self):
        # Prefer scripts/infer.sh if present
        sh = os.path.join(self.repo_dir, "scripts", "infer.sh")
        if os.path.isfile(sh):
            return ("sh", [sh])
        # Else choose python module based on variant
        mapping = {
            "sd15": "src.infer_gsdiff_sd",
            "sd35m": "src.infer_gsdiff_sd3",
            "pas": "src.infer_gsdiff_pas",
            "pixart": "src.infer_gsdiff_pas",
        }
        mod = mapping.get(self.variant, "src.infer_gsdiff_sd")
        return ("python", ["-m", mod])

    def text_to_splat(
        self, prompt: str, seed: int | None = None, timeout_s: float = 28.0
    ) -> Dict[str, object]:
        """
        Runs DiffSplat and returns dict: {"cover": PIL.Image, "ply": bytes|None, "mp4": bytes|None}
        """
        work = tempfile.mkdtemp(prefix="diffsplat_run_")
        out_dir = os.path.join(work, "out")
        os.makedirs(out_dir, exist_ok=True)

        cmdtype, basecmd = self._infer_entry()
        cmd = []
        if cmdtype == "sh":
            cmd = basecmd + [
                "--prompt",
                prompt,
                "--output",
                out_dir,
                "--variant",
                self.variant,
            ]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            run_cwd = ""
        else:
            # python -m src.infer_gsdiff_* --prompt "..." --out out_dir --mp4 --ply
            cmd = (
                ["python"]
                + basecmd
                + ["--prompt", prompt, "--out", out_dir, "--mp4", "--ply"]
            )
            if seed is not None:
                cmd += ["--seed", str(seed)]
            run_cwd = self.repo_dir

        self.logger.info(
            "infer_start",
            extra={"extra": {"cmd": cmd[:3] + ["..."], "run_cwd": run_cwd}},
        )

        done = time_block()
        try:
            subprocess.run(cmd, cwd=run_cwd, check=True, timeout=timeout_s)
            self.logger.info(
                "infer_ok",
                extra={
                    "extra": {
                        "elapsed_s": done(),
                    }
                },
            )
        except subprocess.TimeoutExpired:
            self.logger.error("infer_timeout")
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError("DiffSplat timeout")
        except subprocess.CalledProcessError as e:
            self.logger.error("infer_fail", extra={"extra": {"stderr": e.stderr}})
            shutil.rmtree(work, ignore_errors=True)
            raise RuntimeError(f"DiffSplat failed: {e}")

        # Collect outputs
        cover = None
        # try jpg/png under out/
        for p in glob.glob(os.path.join(out_dir, "*.png")) + glob.glob(
            os.path.join(out_dir, "*.jpg")
        ):
            try:
                cover = Image.open(p).convert("RGB")
                break
            except Exception:
                pass
        if cover is None:
            from PIL import ImageDraw

            cover = Image.new("RGB", (512, 512), (8, 8, 8))
            d = ImageDraw.Draw(cover)
            d.text((16, 16), "no cover from DiffSplat", fill=(220, 220, 220))

        ply_bytes = None
        for ext in ["*.ply", "*.splat"]:
            g = glob.glob(os.path.join(out_dir, ext))
            if g:
                with open(g[0], "rb") as f:
                    ply_bytes = f.read()
                break

        mp4_bytes = None
        g = glob.glob(os.path.join(out_dir, "*.mp4"))
        if g:
            with open(g[0], "rb") as f:
                mp4_bytes = f.read()

        # Cleanup
        shutil.rmtree(work, ignore_errors=True)
        return {"cover": cover, "ply": ply_bytes, "mp4": mp4_bytes}
