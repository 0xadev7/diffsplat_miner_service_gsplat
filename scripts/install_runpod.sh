#!/usr/bin/env bash
set -euo pipefail

# This script prepares the environment on a RunPod pod with PyTorch 2.4 + CUDA 12.4 already installed.
# It installs system deps, clones DiffSplat, applies small patches if needed, and installs Python deps
# (excluding torch). It is idempotent.

echo "[install] Updating apt packages..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  git cmake build-essential ffmpeg libgl1 libglib2.0-0 libeigen3-dev libglm-dev

# Python deps (no torch here, rely on pod)
python -m pip install --upgrade pip
pip install -r requirements.txt

# Clone or update DiffSplat
if [ ! -d "DiffSplat" ]; then
  git clone https://github.com/chenguolin/DiffSplat.git
else
  (cd DiffSplat && git pull --ff-only || true)
fi

# Run upstream setup (does NOT enforce torch version; they note it's flexible)
(cd DiffSplat && bash settings/setup.sh || true)

# Header guard patches (no-op if not present)
for f in \
  DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h \
  DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h; do
  if [ -f "$f" ]; then
    grep -q "<cstdint>" "$f" || sed -i '1i #include <cstdint>' "$f"
  fi
done

# Best-effort build of rasterizer extension to warm caches
if [ -f DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization/setup.py ]; then
  (cd DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization && python setup.py build_ext --inplace || true)
fi

echo "[install] Done."
