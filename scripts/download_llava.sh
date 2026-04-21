#!/usr/bin/env bash
# Download LLaVA-1.5 pretrain data (558k conversations + associated images).
# ~14 GB. Store under data/llava/ so the config paths line up.

set -e
mkdir -p data/llava
cd data/llava

# Conversations file (small).
if [ ! -f llava_pretrain_558k.json ]; then
  echo "[dl] conversations json ..."
  wget -q --show-progress \
    https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json \
    -O llava_pretrain_558k.json
fi

# Image zip (large).
if [ ! -d images ]; then
  echo "[dl] image archive ..."
  wget -q --show-progress \
    https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip \
    -O images.zip
  echo "[dl] unzipping ..."
  if command -v unzip >/dev/null 2>&1; then
    unzip -q images.zip -d images
  else
    echo "[dl] unzip not found, falling back to python zipfile ..."
    python3 -c "
import zipfile, sys
with zipfile.ZipFile('images.zip') as z:
    z.extractall('images')
"
  fi
  rm images.zip
fi

echo "[dl] done. verify:"
ls -la llava_pretrain_558k.json
du -sh images
