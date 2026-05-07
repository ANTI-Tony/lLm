#!/usr/bin/env bash
# Sanity v2: confirm whether mixed-K SFT beats vanilla SFT, or is noise.
# 3 seeds × 2 conditions × 200 eval samples ≈ 2.5h, ~$5 on shared A100.

set -e
cd "$(dirname "$0")/.."

QWEN_PATH=$(find /data_sdg/hibug/models/qwen_ms -name "config.json" -exec dirname {} \; | head -1)
echo "Using Qwen at: $QWEN_PATH"

mkdir -p a2_sanity/results a2_sanity/ckpts

SEEDS="1 2 3"

for seed in $SEEDS; do
    echo ""
    echo "=== seed=$seed: vanilla K=1 SFT ==="
    python a2_sanity/train_loop_sft.py \
        --base_model "$QWEN_PATH" \
        --K 1 --max_samples 1000 --epochs 3 \
        --seed $seed \
        --output a2_sanity/ckpts/v_s${seed}.pt 2>&1 \
        | tee a2_sanity/results/v_s${seed}_sft.log

    echo ""
    echo "=== seed=$seed: mixed-K SFT ==="
    python a2_sanity/train_loop_sft.py \
        --base_model "$QWEN_PATH" \
        --K 4 --mixed_K 1 2 4 \
        --max_samples 1000 --epochs 3 \
        --seed $seed \
        --output a2_sanity/ckpts/m_s${seed}.pt 2>&1 \
        | tee a2_sanity/results/m_s${seed}_sft.log
done

echo ""
echo "=== Eval all 6 checkpoints at K_eval=1, 200 samples ==="
for tag in v_s1 v_s2 v_s3 m_s1 m_s2 m_s3; do
    echo ""
    echo "-- eval $tag --"
    python a2_sanity/eval_loop.py \
        --base_model "$QWEN_PATH" \
        --ckpt a2_sanity/ckpts/${tag}.pt \
        --K_eval 1 \
        --max_samples 200 \
        --output a2_sanity/results/${tag}_eval.json 2>&1 \
        | tee a2_sanity/results/${tag}_eval.log
done

echo ""
echo "=== Decision time ==="
python a2_sanity/decide_v2.py \
    --vanilla a2_sanity/results/v_s1_eval.json \
              a2_sanity/results/v_s2_eval.json \
              a2_sanity/results/v_s3_eval.json \
    --mixed   a2_sanity/results/m_s1_eval.json \
              a2_sanity/results/m_s2_eval.json \
              a2_sanity/results/m_s3_eval.json \
    | tee a2_sanity/results/verdict_v2.txt
