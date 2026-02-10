#!/bin/bash

set -e

cd "$(dirname "$0")"

# Clone dust3r repository (remove existing to ensure clean setup)
rm -rf dust3r
git clone -b dev --recursive https://github.com/camenduru/dust3r

python -m pip install -r requirements.txt

if [[ "$(uname)" == "Linux" ]]; then
  python -m pip install https://github.com/camenduru/wheels/releases/download/colab/curope-0.0.0-cp310-cp310-linux_x86_64.whl
fi

mkdir -p dust3r/checkpoints

WEIGHTS=dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
if [ ! -f "$WEIGHTS" ]; then
  wget https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -P dust3r/checkpoints
fi

echo "Setup complete."