"""
Path C sanity check: does Huginn's recurrent depth help pure text reasoning?

Loads raw base Huginn (no VLM wrapping, no projector), runs GSM8K with
a sweep over num_steps, and writes accuracy + predictions to JSON.

Decision rule:
    * If accuracy rises monotonically or peaks at num_steps > 4 on a
      text reasoning task, Huginn's loops ARE useful for reasoning and
      H1 is a priori plausible — our problem is purely in VLM integration.
    * If accuracy is flat or monotonically decreasing, Huginn's loops
      do not help reasoning at all, and H1 is unlikely to hold even with
      a perfect VLM. Better to pivot to the fallback paper.

Run on A100:
    python huginn_gsm8k.py --num_samples 100
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

torch.backends.cudnn.enabled = False  # same cuDNN workaround


# Few-shot block that matches Huginn's base-LM expectations (no chat tokens).
FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: 50/60 = 5/6 hour. She earned 12 * 5/6 = $10. The answer is 10.

Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
A: He writes 3*2 = 6 pages per friend-batch. Twice a week: 6*2 = 12 pages/week. In a year (52 weeks): 12*52 = 624 pages. The answer is 624.

"""


def extract_answer(text: str) -> str | None:
    """Try to extract the numeric answer from a GSM8K-style completion."""
    # Prefer "The answer is X" pattern.
    m = re.search(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Fallback: #### X pattern (used in gold answers).
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    # Last-resort: last number in the completion.
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def parse_gold(answer_field: str) -> str | None:
    """GSM8K gold answers have the numeric answer after '####'."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if m:
        return m.group(1).replace(",", "")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_steps_list", type=int, nargs="+",
                        default=[4, 16, 64])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output", default="results/huginn_gsm8k.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "GSM8K sweep needs GPU"

    print(f"[info] loading {args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[info] loaded in {time.time() - t0:.1f}s")

    print("[info] loading GSM8K")
    ds = load_dataset("gsm8k", "main", split="test")
    samples = [ds[i] for i in range(min(args.num_samples, len(ds)))]

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None, top_k=None, top_p=None, min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
        use_cache=True,
    )

    results = {"model": args.model, "num_samples": len(samples),
               "few_shot": 3, "per_num_steps": {}}

    for ns in args.num_steps_list:
        print(f"\n=== num_steps={ns} ===")
        correct = 0
        per_item = []
        t_start = time.time()
        for item in tqdm(samples):
            question = item["question"]
            gold = parse_gold(item["answer"])
            prompt = FEWSHOT + f"Q: {question}\nA:"
            ids = tokenizer.encode(prompt, return_tensors="pt",
                                   add_special_tokens=True).to(device)
            with torch.no_grad():
                out = model.generate(ids, gen_config, tokenizer=tokenizer,
                                     num_steps=ns)
            seq = out.sequences if hasattr(out, "sequences") else out
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            # Strip prompt, also trim at next "Q:" to avoid run-on.
            completion = text[len(prompt):] if text.startswith(prompt) else text
            completion = completion.split("Q:")[0].strip()
            pred = extract_answer(completion)
            try:
                ok = (pred is not None and gold is not None
                      and abs(float(pred) - float(gold)) < 1e-3)
            except ValueError:
                ok = False
            correct += int(ok)
            per_item.append({"question": question, "gold": gold,
                             "pred": pred, "completion": completion[:300],
                             "correct": bool(ok)})

        elapsed = time.time() - t_start
        acc = correct / len(samples)
        print(f"  accuracy = {correct}/{len(samples)} = {acc:.3f}   "
              f"elapsed = {elapsed:.1f}s")
        results["per_num_steps"][str(ns)] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(samples),
            "elapsed_seconds": elapsed,
            "samples": per_item,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    summary = {k: v["accuracy"] for k, v in results["per_num_steps"].items()}
    print(json.dumps(summary, indent=2))
    print(f"[info] full results -> {args.output}")

    xs = sorted(int(k) for k in summary)
    ys = [summary[str(x)] for x in xs]
    peak = max(range(len(ys)), key=lambda i: ys[i])
    print(f"\npeak @ num_steps={xs[peak]}  accuracy={ys[peak]:.3f}")
    print(f"step-{xs[0]} baseline: {ys[0]:.3f}  delta: {ys[peak] - ys[0]:+.3f}")
    verdict = "peak at higher depth -> loop helps reasoning" \
        if xs[peak] > xs[0] and ys[peak] - ys[0] >= 0.02 \
        else "flat / monotonic -> loop does not help text reasoning"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
