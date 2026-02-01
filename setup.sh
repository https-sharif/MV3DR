#!/bin/bash

set -e

cd "$(dirname "$0")"

if [ ! -d "dust3r" ]; then
  git clone -b dev --recursive https://github.com/camenduru/dust3r
fi

pip install -r requirements.txt

mkdir -p dust3r/checkpoints

WEIGHTS=dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
if [ ! -f "$WEIGHTS" ]; then
  wget https://huggingface.co/camenduru/dust3r/resolve/main/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
    -P dust3r/checkpoints
fi

echo "Setup complete."