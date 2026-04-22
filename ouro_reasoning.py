"""
Cross-model phenomenon runner for Ouro-1.4B.

Unlike Huginn (num_steps kwarg), Ouro's recurrent depth is controlled via
config.total_ut_steps, which is set at load time and cannot be changed
per call. So for each num_steps we reload the model once and run all
benchmark samples under that setting.

Usage:
    python ouro_reasoning.py --benchmark gsm8k --num_samples 200 \\
        --num_steps_list 4 8 16 32 64

Output format mirrors huginn_reasoning.py so the same analyze_per_sample.py
works on either.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
)

torch.backends.cudnn.enabled = False

# Reuse the benchmark registry from huginn_reasoning so we don't duplicate.
from huginn_reasoning import BENCHMARKS


def _load_ouro(model_name: str, num_steps: int, dtype):
    """Reload Ouro with total_ut_steps=num_steps. Returns (model, tokenizer)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # Ouro's recurrent-depth knob
    config.total_ut_steps = num_steps
    # Force all steps to run (no adaptive exit) so we isolate the depth
    # effect cleanly — otherwise the model might exit early and we would
    # not know whether fewer effective steps were used.
    try:
        config.early_exit_threshold = 1.0
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=dtype, trust_remote_code=True,
    ).cuda().eval()
    return model, tokenizer


def _run_benchmark_at_ns(model, tokenizer, bench, samples, num_steps,
                         gen_config, max_new_tokens):
    correct = 0
    items = []
    t_start = time.time()
    for item in tqdm(samples):
        prompt = bench["prompt"](item)
        gold = bench["gold"](item)
        ids = tokenizer.encode(prompt, return_tensors="pt",
                               add_special_tokens=True).cuda()
        with torch.no_grad():
            out = model.generate(ids, generation_config=gen_config)
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
    return {
        "accuracy": correct / len(samples),
        "correct": correct,
        "total": len(samples),
        "elapsed_seconds": elapsed,
        "samples": items,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--model", default="ByteDance/Ouro-1.4B")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps_list", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/ouro_{args.benchmark}.json"

    bench = BENCHMARKS[args.benchmark]
    max_new = bench["max_new_tokens"]

    print(f"[info] loading benchmark {args.benchmark}")
    ds_args = bench["dataset"]
    if len(ds_args) == 1:
        ds = load_dataset(ds_args[0], split=bench["split"])
    else:
        ds = load_dataset(ds_args[0], ds_args[1], split=bench["split"])
    samples = [ds[i] for i in range(min(args.num_samples, len(ds)))]
    print(f"[info] {len(samples)} samples")

    results = {"model": args.model, "benchmark": args.benchmark,
               "num_samples": len(samples), "max_new_tokens": max_new,
               "per_num_steps": {}}

    for ns in args.num_steps_list:
        print(f"\n=== Reloading Ouro with total_ut_steps={ns} ===")
        t0 = time.time()
        model, tokenizer = _load_ouro(args.model, ns, torch.bfloat16)
        print(f"[info] reloaded in {time.time() - t0:.1f}s")

        gen_config = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None, top_k=None, top_p=None, min_p=None,
            return_dict_in_generate=True,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )

        stats = _run_benchmark_at_ns(
            model, tokenizer, bench, samples, ns, gen_config, max_new)
        print(f"  accuracy = {stats['correct']}/{stats['total']} = "
              f"{stats['accuracy']:.3f}   elapsed = {stats['elapsed_seconds']:.1f}s")
        results["per_num_steps"][str(ns)] = stats

        # Free GPU memory before the next reload.
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        # Save partial progress after each num_steps — this run is long
        # enough that we don't want to lose it to a crash near the end.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[info] partial saved -> {args.output}")

    print("\n=== Summary ===")
    summary = {k: v["accuracy"] for k, v in results["per_num_steps"].items()}
    print(json.dumps(summary, indent=2))
    xs = sorted(int(k) for k in summary)
    ys = [summary[str(x)] for x in xs]
    peak = max(range(len(ys)), key=lambda i: ys[i])
    print(f"\npeak @ num_steps={xs[peak]}  accuracy={ys[peak]:.3f}")
    print(f"step-{xs[0]} baseline: {ys[0]:.3f}  delta: {ys[peak] - ys[0]:+.3f}")


if __name__ == "__main__":
    main()
