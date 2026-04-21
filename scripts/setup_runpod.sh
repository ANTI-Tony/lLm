#!/usr/bin/env bash
# One-shot RunPod setup. Run this once after logging into your A100 pod.
#
#   chmod +x scripts/setup_runpod.sh
#   ./scripts/setup_runpod.sh

set -e

echo "[setup] python version:"; python3 --version
echo "[setup] nvidia-smi:"; nvidia-smi | head -5

pip install --upgrade pip

# Transformers pin matters: Ouro requires <4.56.0; Huginn works with 4.54.1.
pip install -r requirements.txt

echo "[setup] warming HuggingFace cache for Huginn ..."
python3 - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
_ = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
_ = AutoModelForCausalLM.from_pretrained(
    "tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16,
    trust_remote_code=True)
print("huginn ok")
PY

echo "[setup] warming CLIP cache ..."
python3 - <<'PY'
from transformers import CLIPVisionModel, CLIPImageProcessor
_ = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
_ = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
print("clip ok")
PY

echo "[setup] done."
