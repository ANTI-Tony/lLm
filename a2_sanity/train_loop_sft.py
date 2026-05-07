"""
Minimal SFT training for A2 sanity check.

Trains a LoopedLlama on a tiny GSM8K subset (1000 samples, 3 epochs).
Saves checkpoint. Total runtime: ~10-12h on A100-80G for K=4.

Usage:
    python train_loop_sft.py --K 1 --output ckpts/k1.pt --max_samples 1000
    python train_loop_sft.py --K 4 --output ckpts/k4.pt --max_samples 1000
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from looped_llama import LoopedLlama, LoopedLlamaConfig
from data_utils import load_gsm8k

torch.backends.cudnn.enabled = False  # avoid the cuDNN issues we hit before


# ---------- dataset ----------

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Answer: Let's think step by step. "
)


class GSM8KSFTDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_samples=1000,
                 max_seq_length=512):
        ds = load_gsm8k(split)
        self.data = ds[:max_samples]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]  # contains rationale + #### final_number

        prompt = PROMPT_TEMPLATE.format(question=question)
        full = prompt + answer + self.tokenizer.eos_token

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full, add_special_tokens=True)

        input_ids = torch.tensor(full_ids[:self.max_seq_length], dtype=torch.long)
        labels = input_ids.clone()
        # Mask prompt tokens — only learn on answer span.
        labels[:min(len(prompt_ids), len(input_ids))] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id: int):
    max_len = max(b["input_ids"].size(0) for b in batch)

    def _pad(t, pad_val):
        if t.size(0) == max_len:
            return t
        return torch.cat([t, torch.full((max_len - t.size(0),), pad_val,
                                        dtype=t.dtype)], dim=0)

    input_ids = torch.stack([_pad(b["input_ids"], pad_id) for b in batch])
    labels = torch.stack([_pad(b["labels"], -100) for b in batch])
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask}


# ---------- training ----------

def cosine_schedule(optimizer, total_steps, warmup_ratio=0.03):
    warmup = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--K", type=int, required=True,
                   help="loop count during training (ignored if --mixed_K)")
    p.add_argument("--mixed_K", type=int, nargs="+", default=None,
                   help="if set, sample K randomly from this list per micro-batch "
                        "(curriculum / shortcut-consistency style). "
                        "Overrides --K for the actual training pass.")
    p.add_argument("--n_loop_layers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # ---- model ----
    print(f"[info] loading {args.base_model}")
    cfg = LoopedLlamaConfig(
        base_model=args.base_model,
        n_loop_layers=args.n_loop_layers,
        K=args.K,
        input_injection=True,
        injection_scale=0.1,
        loop_layernorm=True,
    )
    model = LoopedLlama(cfg, torch_dtype=torch.bfloat16).to(device)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    print(f"[info] trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")

    # ---- data ----
    ds = GSM8KSFTDataset(tokenizer, split="train",
                         max_samples=args.max_samples,
                         max_seq_length=args.max_seq_length)
    print(f"[info] train samples: {len(ds)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2, drop_last=True,
    )

    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(f"[info] steps/epoch={steps_per_epoch}, total={total_steps}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = cosine_schedule(optim, total_steps, args.warmup_ratio)

    # ---- mixed-K logging ----
    if args.mixed_K:
        print(f"[info] MIXED-K curriculum: K sampled per micro-batch from {args.mixed_K}")
        rng = random.Random(args.seed)
    else:
        print(f"[info] fixed K={args.K} per micro-batch")

    # ---- train loop ----
    optim.zero_grad()
    global_step = 0
    t0 = time.time()
    running = 0.0
    K_seen = []  # for diagnostic histogram
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            K_step = (rng.choice(args.mixed_K) if args.mixed_K else args.K)
            K_seen.append(K_step)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        K=K_step)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % 10 == 0:
                    avg = running / 10 / args.grad_accum
                    elapsed = time.time() - t0
                    lr = sched.get_last_lr()[0]
                    print(f"  ep {epoch} step {global_step}/{total_steps}  "
                          f"loss={avg:.3f}  lr={lr:.2e}  sec={elapsed:.0f}",
                          flush=True)
                    running = 0.0

    # ---- mixed-K diagnostic histogram ----
    if args.mixed_K:
        from collections import Counter
        hist = Counter(K_seen)
        print(f"[info] mixed-K histogram: "
              f"{dict(sorted(hist.items()))}  total={len(K_seen)}")

    # ---- save ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": vars(args),
        "loop_config": cfg.__dict__,
    }, args.output)
    print(f"[info] saved -> {args.output}")


if __name__ == "__main__":
    main()
