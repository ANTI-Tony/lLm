#!/usr/bin/env bash
# One-shot RunPod setup. Run this once after logging into your A100 pod.
#
#   chmod +x scripts/setup_runpod.sh
#   ./scripts/setup_runpod.sh

set -e

echo "[setup] python version:"; python3 --version
echo "[setup] nvidia-smi:"; nvidia-smi | head -5

pip install --upgrade pip

# Huginn's modeling uses torch.nn.attention.flex_attention, which only exists
# in torch>=2.5.0. Pin to 2.5.1 + cu124 wheels because plain "torch>=2.5.0"
# pulls the latest (2.7+/2.8+) whose CUDA build may be too new for the pod's
# driver, silently dropping to CPU. If cu124 doesn't match your driver,
# switch the index-url to cu121.
pip install --force-reinstall --no-deps \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not avail after torch install; try cu121 wheel'; print(f'torch={torch.__version__} device={torch.cuda.get_device_name(0)}')"
python3 -c "from torch.nn.attention.flex_attention import flex_attention; print('flex_attention OK')"

# Transformers pin: Huginn's HuginnDynamicCache conflicts with the property
# refactor of DynamicCache in 4.49+. Force-reinstall 4.48.3 in case the base
# image shipped a newer one.
pip install -r requirements.txt
pip install --force-reinstall --no-deps transformers==4.48.3
python3 -c "import transformers; print(f'transformers={transformers.__version__}')"

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
