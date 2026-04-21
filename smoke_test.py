"""
Phase 0 smoke test (text-only).

Purpose:
    1. Verify Huginn-0125 loads on the rented A100.
    2. Verify num_steps is accepted in forward() and generate().
    3. Measure latency and simple text accuracy at num_steps in {4, 8, 16, 32, 64}.
    4. Confirm that forward(inputs_embeds=..., num_steps=...) works — this is the
       critical path for the LoopedVLM wrapper. If it fails here, the whole plan
       needs to be redesigned before any VLM work.

Run on RunPod A100:
    python smoke_test.py
"""

import argparse
import time
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEFAULT_PROMPTS = [
    "The capital of France is",
    "To solve the equation 2x + 3 = 11, we",
    "The chemical symbol for gold is",
    "In 2008, the financial crisis was primarily caused by",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tomg-group-umd/huginn-0125")
    parser.add_argument("--num_steps_list", type=int, nargs="+",
                        default=[4, 8, 16, 32, 64])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output", default="results/smoke_test.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — this test is meant for A100.")

    print(f"[info] loading {args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[info] load time: {time.time() - t0:.1f}s")

    # Check 1: vanilla forward with num_steps
    print("\n[check-1] forward(input_ids, num_steps=...)")
    input_ids = tokenizer.encode("Hello world", return_tensors="pt",
                                 add_special_tokens=True).to(device)
    try:
        with torch.no_grad():
            out = model(input_ids, num_steps=8)
        logits_shape = tuple(out.logits.shape) if hasattr(out, "logits") else None
        print(f"        OK. logits shape = {logits_shape}")
    except Exception as e:
        print(f"        FAIL: {e}")
        sys.exit(1)

    # Check 2: forward with inputs_embeds (CRITICAL for VLM wrapper)
    print("\n[check-2] forward(inputs_embeds=..., num_steps=...)  (needed for VLM)")
    try:
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        with torch.no_grad():
            out = model(inputs_embeds=inputs_embeds, num_steps=8)
        print(f"        OK. logits shape = {tuple(out.logits.shape)}")
        vlm_viable = True
    except Exception as e:
        print(f"        FAIL (VLM path broken): {e}")
        print("        -> will need to patch modeling code or pre-concat embeddings "
              "into a custom wrapper.")
        vlm_viable = False

    # Check 3: generation with num_steps sweep
    print("\n[check-3] generate() sweep over num_steps")
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None, top_k=None, top_p=None, min_p=None,
        return_dict_in_generate=True,
        eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
        use_cache=True,
    )

    results = {"model": args.model, "vlm_inputs_embeds_ok": vlm_viable,
               "prompts": DEFAULT_PROMPTS, "per_num_steps": {}}

    for num_steps in args.num_steps_list:
        per_prompt = []
        tot_tok = 0
        t_start = time.time()
        for prompt in DEFAULT_PROMPTS:
            ids = tokenizer.encode(prompt, return_tensors="pt",
                                   add_special_tokens=True).to(device)
            with torch.no_grad():
                gen = model.generate(ids, gen_config, tokenizer=tokenizer,
                                     num_steps=num_steps)
            seq = gen.sequences if hasattr(gen, "sequences") else gen
            text = tokenizer.decode(seq[0], skip_special_tokens=True)
            new_tok = seq.shape[-1] - ids.shape[-1]
            tot_tok += max(new_tok, 0)
            per_prompt.append({"prompt": prompt, "completion": text})
        elapsed = time.time() - t_start
        tok_per_s = tot_tok / elapsed if elapsed > 0 else 0.0
        print(f"  num_steps={num_steps:>3d}  time={elapsed:6.1f}s  "
              f"tok/s={tok_per_s:5.1f}")
        results["per_num_steps"][str(num_steps)] = {
            "wall_seconds": elapsed,
            "tokens_per_second": tok_per_s,
            "completions": per_prompt,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[info] saved -> {args.output}")

    if not vlm_viable:
        print("\n[IMPORTANT] inputs_embeds path failed. Before running any VLM code, "
              "inspect modeling_huginn to find the forward signature.")
        sys.exit(2)


if __name__ == "__main__":
    main()
