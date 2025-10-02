#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "${TORCH_HOME:-$HOME/.cache/torch}"
echo "[prepare_models] Using HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
