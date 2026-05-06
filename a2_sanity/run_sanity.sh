#!/usr/bin/env bash
# A2 sanity check — 3-day decision protocol.
#
# Uses Qwen2.5-1.5B (downloaded via ModelScope to /data_sdg/hibug/models/qwen_ms).
# This avoids the gated Llama and the unstable HF download path.
#
# Run from repo root:
#   bash a2_sanity/run_sanity.sh
#
# Prereq env vars:
#   conda activate lLm
#   clashon       # only needed for GSM8K dataset download
#   export HF_HOME=/data_sdg/$USER/hf_cache
#   export CUDA_VISIBLE_DEVICES=0  # GPU 0 is currently free; GPU 1 is occupied

set -e
cd "$(dirname "$0")/.."  # cd to repo root

# Resolve Qwen local path (modelscope substitutes . with ___ in dirnames)
QWEN_PATH=$(find /data_sdg/hibug/models/qwen_ms -name "config.json" -exec dirname {} \; | head -1)
if [ -z "$QWEN_PATH" ]; then
    echo "ERROR: Qwen2.5-1.5B not found under /data_sdg/hibug/models/qwen_ms"
    echo "Run: python -c 'from modelscope import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-1.5B\", cache_dir=\"/data_sdg/hibug/models/qwen_ms\")'"
    exit 1
fi
echo "Using Qwen at: $QWEN_PATH"

mkdir -p a2_sanity/results a2_sanity/ckpts

echo ""
echo "=== Day 1: Zero-shot baseline (no training, ~30 min) ==="
echo "Verifies infra works. Expect K=1 reasonable, K>1 likely worse."
python a2_sanity/eval_loop.py \
    --base_model "$QWEN_PATH" \
    --K_eval 1 2 4 \
    --max_samples 200 \
    --output a2_sanity/results/zeroshot.json

echo ""
echo "=== Day 2a: K=1 SFT baseline (~3-4h on A100) ==="
python a2_sanity/train_loop_sft.py \
    --base_model "$QWEN_PATH" \
    --K 1 \
    --max_samples 1000 \
    --epochs 3 \
    --output a2_sanity/ckpts/k1.pt

echo ""
echo "=== Day 2b: K=4 loop SFT (~10-12h on A100) ==="
python a2_sanity/train_loop_sft.py \
    --base_model "$QWEN_PATH" \
    --K 4 \
    --max_samples 1000 \
    --epochs 3 \
    --output a2_sanity/ckpts/k4.pt

echo ""
echo "=== Day 3a: Eval K=1 trained ckpt at K_eval={1,2,4} ==="
python a2_sanity/eval_loop.py \
    --base_model "$QWEN_PATH" \
    --ckpt a2_sanity/ckpts/k1.pt \
    --K_eval 1 2 4 \
    --max_samples 200 \
    --output a2_sanity/results/k1_trained.json

echo ""
echo "=== Day 3b: Eval K=4 trained ckpt at K_eval={1,2,4} ==="
python a2_sanity/eval_loop.py \
    --base_model "$QWEN_PATH" \
    --ckpt a2_sanity/ckpts/k4.pt \
    --K_eval 1 2 4 \
    --max_samples 200 \
    --output a2_sanity/results/k4_trained.json

echo ""
echo "=== Decision time ==="
python a2_sanity/decide.py \
    --zeroshot a2_sanity/results/zeroshot.json \
    --k1_trained a2_sanity/results/k1_trained.json \
    --k4_trained a2_sanity/results/k4_trained.json
