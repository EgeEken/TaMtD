#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ffmpeg \
  libavif-bin \
  libavif-dev \
  libjpeg-turbo-progs \
  libjpeg-turbo8-dev \
  libjxl-dev \
  libjxl-tools \
  libwebp-dev \
  pkg-config \
  pipx \
  python3.12 \
  python3.12-dev \
  python3.12-venv \
  webp

pipx install "pdm==2.26.9"
export PATH="$HOME/.local/bin:$PATH"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pdm sync --dev -G staged-jpeg

echo "Environment ready. Set DATA_ROOT, CACHE_ROOT, RESULTS_ROOT, TEMP_ROOT, PBC_ROOT, and CUDA_DEVICE before running experiments."
