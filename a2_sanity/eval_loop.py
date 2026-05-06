"""
GSM8K eval for LoopedLlama checkpoints.

Evaluates a trained K=N model at multiple K_eval values to see how loop
count at inference interacts with training. Greedy decode, max 128 new
tokens, parses last number from completion.

Usage:
    # Eval no-train baseline at K=1, 2, 4 (zero-shot, no checkpoint)
    python eval_loop.py --base_model meta-llama/Llama-3.2-1B \\
        --K_eval 1 2 4 --max_samples 200 \\
        --output results/zeroshot.json

    # Eval trained checkpoint
    python eval_loop.py --ckpt ckpts/k4.pt --K_eval 1 2 4 \\
        --max_samples 200 --output results/k4_trained.json
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from looped_llama import LoopedLlama, LoopedLlamaConfig
from data_utils import load_gsm8k

torch.backends.cudnn.enabled = False


PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: Let's think step by step. "
)


def extract_answer(text: str):
    # Try "#### X" pattern first
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Then last number
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def parse_gold(answer_field: str):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return m.group(1).replace(",", "") if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--ckpt", default=None,
                   help="path to .pt checkpoint; if None, use untrained base")
    p.add_argument("--K_eval", type=int, nargs="+", required=True)
    p.add_argument("--n_loop_layers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = LoopedLlamaConfig(
        base_model=args.base_model,
        n_loop_layers=args.n_loop_layers,
        K=1,  # placeholder; we override per-call in generate_greedy
        input_injection=True,
        injection_scale=0.1,
        loop_layernorm=True,
    )
    model = LoopedLlama(cfg, torch_dtype=torch.bfloat16).to(device)

    if args.ckpt:
        print(f"[info] loading checkpoint {args.ckpt}")
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["state_dict"])
    else:
        print("[info] no checkpoint — using untrained base + loop wrapper")

    model.eval()

    # ---- data ----
    ds = load_gsm8k("test")
    samples = ds[: args.max_samples]

    eos_id = tokenizer.eos_token_id

    results = {"base_model": args.base_model, "ckpt": args.ckpt,
               "n_loop_layers": args.n_loop_layers,
               "num_samples": len(samples), "per_K": {}}

    for K in args.K_eval:
        print(f"\n=== K_eval={K} ===")
        correct = 0
        items = []
        t0 = time.time()
        for item in tqdm(samples):
            prompt = PROMPT_TEMPLATE.format(question=item["question"])
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            seq = model.generate_greedy(ids, max_new_tokens=args.max_new_tokens,
                                        K=K, eos_token_id=eos_id)
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            completion = text[len(prompt):] if text.startswith(prompt) else text
            completion = completion.split("Question:")[0].strip()
            pred = extract_answer(completion)
            gold = parse_gold(item["answer"])
            ok = (pred is not None and gold is not None
                  and float(pred) == float(gold))
            correct += int(ok)
            items.append({"question": item["question"][:100],
                          "gold": gold, "pred": pred,
                          "completion": completion[:200], "correct": bool(ok)})
        elapsed = time.time() - t0
        acc = correct / len(samples)
        print(f"  acc = {correct}/{len(samples)} = {acc:.3f}  "
              f"elapsed={elapsed:.1f}s")
        results["per_K"][str(K)] = {"accuracy": acc, "correct": correct,
                                    "total": len(samples),
                                    "elapsed_seconds": elapsed,
                                    "samples": items}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    summary = {k: v["accuracy"] for k, v in results["per_K"].items()}
    print(json.dumps(summary, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
