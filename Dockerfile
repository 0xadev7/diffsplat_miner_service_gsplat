FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace
RUN pip3 install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/chenguolin/DiffSplat.git /workspace/DiffSplat || true \
 && python3 /workspace/DiffSplat/download_ckpt.py || true

EXPOSE 8093
CMD ["python3", "-m", "app.server"]