"""
MATH-500 eval for LoopedLlama checkpoints (cross-task validation).

Same model + wrapper as eval_loop.py, but on MATH-500 with LaTeX-aware
answer parsing. We use a tolerant equivalence check: extract the last
\\boxed{...} from the completion, normalize whitespace + simple LaTeX,
and compare to the gold answer also normalized.

Usage:
    python eval_math.py --base_model <qwen> --ckpt ckpts/v_s1.pt \\
        --K_eval 1 --max_samples 100 --output results/v_s1_math.json
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
from data_utils import load_math500

torch.backends.cudnn.enabled = False


PROMPT_TEMPLATE = (
    "Problem: {question}\n"
    "Solution: Let's solve step by step. "
)


# ---------- answer parsing ----------

def _extract_boxed(text: str):
    """Find the LAST \\boxed{...} content in text, balancing braces."""
    out = None
    i = 0
    while True:
        idx = text.find(r"\boxed", i)
        if idx == -1:
            break
        j = idx + len(r"\boxed")
        # skip optional whitespace + opening brace
        while j < len(text) and text[j] in " \t":
            j += 1
        if j >= len(text) or text[j] != "{":
            i = idx + 1
            continue
        # balance braces
        depth = 1
        j += 1
        start = j
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            if depth > 0:
                j += 1
        if depth == 0:
            out = text[start:j]
        i = j + 1
    return out


def _normalize(s: str) -> str:
    """Loose LaTeX normalization for equivalence checking."""
    if s is None:
        return ""
    s = s.strip()
    # remove enclosing $ ... $ and \( \), \[ \]
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"^\\\[|\\\]$", "", s).strip()
    # remove \left \right
    s = s.replace(r"\left", "").replace(r"\right", "")
    # remove \! \, \; \: \quad \qquad
    for tok in [r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad"]:
        s = s.replace(tok, "")
    # whitespace
    s = re.sub(r"\s+", "", s)
    # \dfrac -> \frac
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return s


def extract_pred(completion: str):
    boxed = _extract_boxed(completion)
    if boxed is not None:
        return boxed
    # fallback: last math expression in $...$
    m = list(re.finditer(r"\$([^$]+)\$", completion))
    if m:
        return m[-1].group(1)
    # fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", completion)
    return nums[-1] if nums else None


def is_correct(pred, gold) -> bool:
    if pred is None or gold is None:
        return False
    a = _normalize(pred)
    b = _normalize(gold)
    if a == b:
        return True
    # numeric float fallback
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (ValueError, TypeError):
        return False


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--K_eval", type=int, nargs="+", default=[1])
    p.add_argument("--n_loop_layers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = LoopedLlamaConfig(
        base_model=args.base_model,
        n_loop_layers=args.n_loop_layers,
        K=1,
        input_injection=True,
        injection_scale=0.1,
        loop_layernorm=True,
    )
    model = LoopedLlama(cfg, torch_dtype=torch.bfloat16).to(device)

    print(f"[info] loading {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"])
    model.eval()

    samples = load_math500("test")[: args.max_samples]
    eos_id = tokenizer.eos_token_id

    results = {"base_model": args.base_model, "ckpt": args.ckpt,
               "benchmark": "MATH-500",
               "num_samples": len(samples), "per_K": {}}

    for K in args.K_eval:
        print(f"\n=== K_eval={K} ===")
        correct = 0
        items = []
        t0 = time.time()
        for s in tqdm(samples):
            prompt = PROMPT_TEMPLATE.format(question=s["question"])
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            seq = model.generate_greedy(ids,
                                        max_new_tokens=args.max_new_tokens,
                                        K=K, eos_token_id=eos_id)
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            completion = text[len(prompt):] if text.startswith(prompt) else text
            completion = completion.split("Problem:")[0].strip()
            pred = extract_pred(completion)
            ok = is_correct(pred, s["answer"])
            correct += int(ok)
            items.append({"gold": s["answer"], "pred": pred,
                          "level": s["level"], "subject": s["subject"],
                          "completion": completion[:300],
                          "correct": bool(ok)})
        elapsed = time.time() - t0
        acc = correct / len(samples)
        print(f"  acc = {correct}/{len(samples)} = {acc:.3f}  elapsed={elapsed:.1f}s")
        results["per_K"][str(K)] = {"accuracy": acc, "correct": correct,
                                    "total": len(samples),
                                    "elapsed_seconds": elapsed,
                                    "samples": items}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
