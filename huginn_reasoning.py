"""
Path A1 reasoning benchmark runner for base Huginn.

Supports multiple reasoning benchmarks through a benchmark registry. Each
entry specifies dataset path, prompt-building logic, and answer-extraction
heuristics.

Usage:
    python huginn_reasoning.py --benchmark gsm8k --num_samples 200 \
        --num_steps_list 4 8 16 32 64

    python huginn_reasoning.py --benchmark arc_challenge --num_samples 150 \
        --num_steps_list 4 16 64

    python huginn_reasoning.py --benchmark math --num_samples 100 \
        --num_steps_list 4 16 64

Output: results/huginn_<bench>.json with per-num_steps accuracy + samples.
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

torch.backends.cudnn.enabled = False


# =============================== Benchmarks ===============================

GSM8K_FEWSHOT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. The answer is 72.

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: 50/60 = 5/6 hour. She earned 12 * 5/6 = $10. The answer is 10.

Q: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
A: 3*2 = 6 pages per friend-batch. Twice a week: 6*2 = 12 pages/week. Over 52 weeks: 12*52 = 624 pages. The answer is 624.

"""

ARC_FEWSHOT = """Q: Which of the following is an example of a physical change?
Options: A) iron rusting  B) wood burning  C) ice melting  D) food digesting
A: Ice melting changes state without altering chemical identity. That is a physical change. The answer is C.

Q: What force pulls objects toward the center of the Earth?
Options: A) friction  B) gravity  C) magnetism  D) electricity
A: Gravity is the force that pulls objects toward planet centers. The answer is B.

Q: Which organelle in a plant cell captures sunlight for photosynthesis?
Options: A) nucleus  B) mitochondria  C) chloroplast  D) ribosome
A: Chloroplasts contain chlorophyll and perform photosynthesis. The answer is C.

"""

MATH_FEWSHOT = """Q: If f(x) = 2x + 3 and g(x) = x^2 - 1, find f(g(2)).
A: g(2) = 2^2 - 1 = 3. f(3) = 2*3 + 3 = 9. The answer is 9.

Q: Solve for x: 3(x - 4) = 2x + 5.
A: 3x - 12 = 2x + 5. 3x - 2x = 5 + 12. x = 17. The answer is 17.

Q: Find the area of a triangle with base 10 and height 6.
A: Area = (1/2) * base * height = (1/2) * 10 * 6 = 30. The answer is 30.

"""


def _extract_number(text: str) -> str | None:
    m = re.search(r"[Tt]he answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_letter(text: str) -> str | None:
    m = re.search(r"[Tt]he answer is\s*([A-D])\b", text)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return None


def _gsm8k_prompt(item):
    return GSM8K_FEWSHOT + f"Q: {item['question']}\nA:"


def _gsm8k_gold(item):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", item["answer"])
    return m.group(1).replace(",", "") if m else None


def _gsm8k_score(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return False


def _arc_prompt(item):
    choices = item["choices"]
    labels = choices["label"]
    texts = choices["text"]
    opts = "  ".join(f"{l}) {t}" for l, t in zip(labels, texts))
    return (ARC_FEWSHOT
            + f"Q: {item['question']}\nOptions: {opts}\nA:")


def _arc_gold(item):
    return str(item["answerKey"]).strip().upper()


def _arc_score(pred, gold):
    if pred is None or gold is None:
        return False
    return pred == gold


def _math_prompt(item):
    return MATH_FEWSHOT + f"Q: {item['problem']}\nA:"


def _math_gold(item):
    # MATH dataset has 'solution' with boxed answer. Extract.
    sol = item["solution"]
    m = re.search(r"\\boxed\{([^}]+)\}", sol)
    if not m:
        return None
    # Try to parse as number (strip LaTeX).
    ans = m.group(1).strip()
    num = re.sub(r"[^\d\.\-/]", "", ans)
    return num if num else ans


def _math_score(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        # Loose textual match.
        return pred.strip() == gold.strip()


BENCHMARKS = {
    "gsm8k": {
        "dataset": ("gsm8k", "main"),
        "split": "test",
        "prompt": _gsm8k_prompt,
        "gold": _gsm8k_gold,
        "extract": _extract_number,
        "score": _gsm8k_score,
        "max_new_tokens": 128,
    },
    "arc_challenge": {
        "dataset": ("allenai/ai2_arc", "ARC-Challenge"),
        "split": "test",
        "prompt": _arc_prompt,
        "gold": _arc_gold,
        "extract": _extract_letter,
        "score": _arc_score,
        "max_new_tokens": 80,
    },
    "math": {
        "dataset": ("hendrycks/competition_math",),
        "split": "test",
        "prompt": _math_prompt,
        "gold": _math_gold,
        "extract": _extract_number,
        "score": _math_score,
        "max_new_tokens": 192,
    },
}


# =============================== Runner ===============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps_list", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/huginn_{args.benchmark}.json"

    bench = BENCHMARKS[args.benchmark]
    max_new = bench["max_new_tokens"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Need GPU"

    print(f"[info] loading {args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[info] loaded in {time.time() - t0:.1f}s")

    print(f"[info] loading benchmark {args.benchmark}")
    ds_args = bench["dataset"]
    if len(ds_args) == 1:
        ds = load_dataset(ds_args[0], split=bench["split"])
    else:
        ds = load_dataset(ds_args[0], ds_args[1], split=bench["split"])
    samples = [ds[i] for i in range(min(args.num_samples, len(ds)))]

    gen_config = GenerationConfig(
        max_new_tokens=max_new,
        do_sample=False,
        temperature=None, top_k=None, top_p=None, min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
        use_cache=True,
    )

    results = {"model": args.model, "benchmark": args.benchmark,
               "num_samples": len(samples), "max_new_tokens": max_new,
               "per_num_steps": {}}

    for ns in args.num_steps_list:
        print(f"\n=== num_steps={ns} ===")
        correct = 0
        items = []
        t_start = time.time()
        for item in tqdm(samples):
            prompt = bench["prompt"](item)
            gold = bench["gold"](item)
            ids = tokenizer.encode(prompt, return_tensors="pt",
                                   add_special_tokens=True).to(device)
            with torch.no_grad():
                out = model.generate(ids, gen_config, tokenizer=tokenizer,
                                     num_steps=ns)
            seq = out.sequences if hasattr(out, "sequences") else out
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            completion = text[len(prompt):] if text.startswith(prompt) else text
            completion = completion.split("Q:")[0].strip()
            pred = bench["extract"](completion)
            ok = bench["score"](pred, gold)
            correct += int(ok)
            items.append({"gold": gold, "pred": pred,
                          "completion": completion[:300],
                          "correct": bool(ok)})

        elapsed = time.time() - t_start
        acc = correct / len(samples)
        print(f"  accuracy = {correct}/{len(samples)} = {acc:.3f}   "
              f"elapsed = {elapsed:.1f}s")
        results["per_num_steps"][str(ns)] = {
            "accuracy": acc, "correct": correct, "total": len(samples),
            "elapsed_seconds": elapsed, "samples": items,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    summary = {k: v["accuracy"] for k, v in results["per_num_steps"].items()}
    print(json.dumps(summary, indent=2))

    xs = sorted(int(k) for k in summary)
    ys = [summary[str(x)] for x in xs]
    peak = max(range(len(ys)), key=lambda i: ys[i])
    print(f"\npeak @ num_steps={xs[peak]}  accuracy={ys[peak]:.3f}")
    print(f"step-{xs[0]} baseline: {ys[0]:.3f}  delta: {ys[peak] - ys[0]:+.3f}")
    print(f"[info] full results -> {args.output}")


if __name__ == "__main__":
    main()
