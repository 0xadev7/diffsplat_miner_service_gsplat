#!/usr/bin/env bash
set -e

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda not found. Please install Miniconda or Anaconda."
  exit 1
fi

if ! conda env list | grep -q "three-gen-mining"; then
  echo "[+] Creating conda env three-gen-mining ..."
  conda env create -f environment.yml
fi

echo "[+] Activating env ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate three-gen-mining

echo "[+] Ensuring pip deps ..."
pip install -r requirements.txt --upgrade

# Pull DiffSplat as sibling
if [ ! -d "../DiffSplat" ]; then
  echo "[+] Cloning DiffSplat ..."
  git clone https://github.com/chenguolin/DiffSplat.git ../DiffSplat
fi

echo "[+] Installing DiffSplat deps & ckpts ..."
pushd ../DiffSplat >/dev/null
pip install -r requirements.txt || true
python ./download_ckpt.py || true
popd >/dev/null

echo "[+] Setup done. To run: conda activate three-gen-mining && python -m app.server"