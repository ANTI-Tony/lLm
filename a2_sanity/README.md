# A2 Sanity Check — Latent Iteration Retrofit

## What this tests

Can a dense pretrained LLM be taught **latent iteration** (Mythos / Huginn /
Ouro style loops) via simple SFT post-training?

Concretely: take Llama-3.2-1B, designate the last 4 layers as a "loop block",
apply them K=4 times during forward, and see if a small SFT run lets the
model use those iterations productively on GSM8K.

If yes → A2 is viable, scale up to full paper.
If no → save weeks. Pivot or kill.

## 3-day cost

| step | time | $ |
|------|------|---|
| Day 1: zero-shot baseline | 30 min | ~$1 |
| Day 2a: K=1 SFT | 3-4h | ~$8 |
| Day 2b: K=4 SFT | 10-12h | ~$25 |
| Day 3: eval × 2 ckpts | 1h | ~$3 |
| **Total** | **~14-18h** | **~$40** |

## Prerequisites

```bash
conda activate lLm
clashon                              # 国内代理
export HF_HOME=/data_sdg/$USER/hf_cache
export CUDA_VISIBLE_DEVICES=1
huggingface-cli login                # 需要 access Llama-3.2-1B (gated repo)
```

`Llama-3.2-1B` is a gated model — you need to:
1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B
2. Accept the license
3. Get an HF token from settings → Access Tokens
4. Run `huggingface-cli login` and paste it

If you can't access Llama, fall back to:
- `Qwen/Qwen2.5-1.5B`  (no gating, similar size)
- `microsoft/Phi-3-mini-4k-instruct`  (3.8B, larger)

## Run it

```bash
cd ~/Tony_wjb/lLm
git pull
tmux new -s sanity     # use tmux so SSH disconnect doesn't kill it
bash a2_sanity/run_sanity.sh 2>&1 | tee a2_sanity/results/run.log
# Ctrl+B D to detach
```

Reattach later: `tmux attach -t sanity`
Watch from outside: `tail -f a2_sanity/results/run.log`

## Files

- `looped_llama.py` — wrapper applying last-N-layers K times
- `train_loop_sft.py` — SFT on GSM8K
- `eval_loop.py` — GSM8K eval, sweeps K_eval
- `decide.py` — reads results, prints GO / STOP / UNCLEAR
- `run_sanity.sh` — orchestrates Day 1-3
