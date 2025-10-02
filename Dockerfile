FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential cmake \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg libgl1 libglib2.0-0 libeigen3-dev libglm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace

# PyTorch for CUDA 12.4 (adjust if needed)
RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Clone DiffSplat and set up
RUN git clone --depth=1 https://github.com/chenguolin/DiffSplat.git && \
    cd DiffSplat && \
    bash settings/setup.sh || true

# Patch frequent header issues (no-op if paths differ)
RUN bash -lc " \
    set -e; \
    for f in \
      DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h \
      DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h; do \
        [ -f $f ] && sed -i '1i #include <cstdint>' $f || true; \
      done"

# Optional build of CUDA ext (best effort)
RUN bash -lc "cd DiffSplat || exit 0; python - <<'PY'\
import os, subprocess, sys\
ext='extensions/RaDe-GS/submodules/diff-gaussian-rasterization'\
setup=os.path.join(ext,'setup.py')\
sys.exit(0) if not os.path.isfile(setup) else subprocess.call([sys.executable, setup, 'build_ext', '--inplace'])\
PY"

ENV HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    UVICORN_WORKERS=1

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./README.md

EXPOSE 8093
CMD ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", "8093"]
