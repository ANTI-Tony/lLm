#!/usr/bin/env bash
# Sanity v3: cross-task validation on MATH-500.
# Re-uses the 6 ckpts trained in sanity v2 — no retraining needed.
# 6 ckpts × 100 MATH-500 samples × ~10s each ≈ 1.5h, ~$3.

set -e
cd "$(dirname "$0")/.."

QWEN_PATH=$(find /data_sdg/hibug/models/qwen_ms -name "config.json" -exec dirname {} \; | head -1)
echo "Using Qwen at: $QWEN_PATH"

mkdir -p a2_sanity/results

for tag in v_s1 v_s2 v_s3 m_s1 m_s2 m_s3; do
    echo ""
    echo "-- MATH eval $tag --"
    python a2_sanity/eval_math.py \
        --base_model "$QWEN_PATH" \
        --ckpt a2_sanity/ckpts/${tag}.pt \
        --K_eval 1 \
        --max_samples 100 \
        --output a2_sanity/results/${tag}_math.json 2>&1 \
        | tee a2_sanity/results/${tag}_math.log
done

echo ""
echo "=== Cross-task verdict ==="
python a2_sanity/decide_v3.py \
    --gsm_vanilla a2_sanity/results/v_s1_eval.json \
                  a2_sanity/results/v_s2_eval.json \
                  a2_sanity/results/v_s3_eval.json \
    --gsm_mixed   a2_sanity/results/m_s1_eval.json \
                  a2_sanity/results/m_s2_eval.json \
                  a2_sanity/results/m_s3_eval.json \
    --math_vanilla a2_sanity/results/v_s1_math.json \
                   a2_sanity/results/v_s2_math.json \
                   a2_sanity/results/v_s3_math.json \
    --math_mixed   a2_sanity/results/m_s1_math.json \
                   a2_sanity/results/m_s2_math.json \
                   a2_sanity/results/m_s3_math.json \
    | tee a2_sanity/results/verdict_v3.txt
