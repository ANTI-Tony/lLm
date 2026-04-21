#!/usr/bin/env bash
# Phase 0 end-to-end script. Assumes setup_runpod.sh + download_llava.sh
# have already been run.
#
#   bash scripts/run_phase0.sh

set -e

echo "=== [step 1/5] smoke test ==="
python3 smoke_test.py --output results/smoke_test.json

echo "=== [step 2/5] projector alignment training ==="
python3 -m src.train.train_projector --config configs/huginn_vlm.yaml

CKPT=checkpoints/projector_huginn/projector_final.pt

echo "=== [step 3/5] eval MMMU ==="
python3 -m src.eval.eval_mmmu --config configs/huginn_vlm.yaml \
    --projector "$CKPT" --output results/mmmu.json

echo "=== [step 4/5] eval MathVista ==="
python3 -m src.eval.eval_mathvista --config configs/huginn_vlm.yaml \
    --projector "$CKPT" --output results/mathvista.json

echo "=== [step 5/5] eval ScienceQA ==="
python3 -m src.eval.eval_scienceqa --config configs/huginn_vlm.yaml \
    --projector "$CKPT" --output results/scienceqa.json

echo "=== plotting ==="
python3 plot_results.py \
    --inputs results/mmmu.json results/mathvista.json results/scienceqa.json \
    --labels MMMU MathVista ScienceQA \
    --out results/phase0_curve.png

echo "=== DONE. Review results/phase0_curve.png and results/phase0_curve.decision.txt ==="
