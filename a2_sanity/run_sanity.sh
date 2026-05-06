#!/usr/bin/env bash
# A2 sanity check — 3-day decision protocol.
#
# Run from repo root:
#   bash a2_sanity/run_sanity.sh
#
# Prereq env vars (set in your shell or ~/.bashrc):
#   conda activate lLm
#   clashon
#   export HF_HOME=/data_sdg/$USER/hf_cache
#   export CUDA_VISIBLE_DEVICES=1

set -e
cd "$(dirname "$0")/.."  # cd to repo root

mkdir -p a2_sanity/results a2_sanity/ckpts

echo ""
echo "=== Day 1: Zero-shot baseline (no training, ~30 min) ==="
echo "Verifies infra works. Expect K=1 reasonable, K>1 likely worse."
python a2_sanity/eval_loop.py \
    --base_model meta-llama/Llama-3.2-1B \
    --K_eval 1 2 4 \
    --max_samples 200 \
    --output a2_sanity/results/zeroshot.json

echo ""
echo "=== Day 2a: K=1 SFT baseline (~3-4h on A100) ==="
python a2_sanity/train_loop_sft.py \
    --base_model meta-llama/Llama-3.2-1B \
    --K 1 \
    --max_samples 1000 \
    --epochs 3 \
    --output a2_sanity/ckpts/k1.pt

echo ""
echo "=== Day 2b: K=4 loop SFT (~10-12h on A100) ==="
python a2_sanity/train_loop_sft.py \
    --base_model meta-llama/Llama-3.2-1B \
    --K 4 \
    --max_samples 1000 \
    --epochs 3 \
    --output a2_sanity/ckpts/k4.pt

echo ""
echo "=== Day 3a: Eval K=1 trained ckpt at K_eval={1,2,4} ==="
python a2_sanity/eval_loop.py \
    --ckpt a2_sanity/ckpts/k1.pt \
    --K_eval 1 2 4 \
    --max_samples 200 \
    --output a2_sanity/results/k1_trained.json

echo ""
echo "=== Day 3b: Eval K=4 trained ckpt at K_eval={1,2,4} ==="
python a2_sanity/eval_loop.py \
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
