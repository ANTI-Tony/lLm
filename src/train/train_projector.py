"""
Projector alignment training.

Only the projector is trained. CLIP and the recurrent-depth LLM are frozen.
Loss: next-token cross-entropy on the ASSISTANT portion only.

Usage:
    accelerate launch -m src.train.train_projector --config configs/huginn_vlm.yaml
"""

import argparse
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Disable cuDNN autotune: on some RunPod cu121 images, cuDNN's algorithm
# benchmarking raises CUDNN_STATUS_NOT_INITIALIZED on the first large Conv2d
# (CLIP's 14x14 patch embedding). Deterministic algo path avoids it at
# negligible cost — we only have one Conv2d in the whole pipeline.
torch.backends.cudnn.benchmark = False

from src.model.looped_vlm import LoopedVLM, LoopedVLMConfig
from src.data.llava_dataset import LlavaPretrainDataset, collate_llava


def _get_num_image_patches(clip_name: str) -> int:
    """ViT-L/14-336: 336/14 = 24 -> 24*24 = 576 patches (no CLS)."""
    table = {
        "openai/clip-vit-large-patch14-336": 576,
        "openai/clip-vit-large-patch14": 256,
        "openai/clip-vit-base-patch16": 196,
    }
    return table.get(clip_name, 576)


def build_scheduler(optimizer, num_training_steps, warmup_ratio):
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"]["seed"])

    model_cfg = LoopedVLMConfig(
        llm_name=cfg["model"]["llm_name"],
        vision_encoder=cfg["model"]["vision_encoder"],
        projector_type=cfg["model"]["projector_type"],
        freeze_llm=cfg["model"]["freeze_llm"],
        freeze_vision=cfg["model"]["freeze_vision"],
    )
    vlm = LoopedVLM(model_cfg, torch_dtype=torch.bfloat16).to(device)

    num_patches = _get_num_image_patches(cfg["model"]["vision_encoder"])

    dataset = LlavaPretrainDataset(
        data_path=cfg["train"]["data_path"],
        image_folder=cfg["train"]["image_folder"],
        tokenizer=vlm.tokenizer,
        image_processor=vlm.image_processor,
        image_token=vlm.cfg.image_placeholder,
        image_token_id=vlm.image_token_id,
        num_image_patches=num_patches,
        max_seq_length=cfg["train"]["max_seq_length"],
        max_samples=cfg["train"].get("max_samples"),
    )
    pad_id = vlm.tokenizer.pad_token_id or 65509
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["per_device_train_batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["dataloader_num_workers"],
        collate_fn=partial(collate_llava, pad_token_id=pad_id),
        drop_last=True,
        pin_memory=True,
    )

    grad_accum = cfg["train"]["gradient_accumulation_steps"]
    epochs = cfg["train"]["num_train_epochs"]
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * epochs

    optim = AdamW(vlm.trainable_parameters(),
                  lr=cfg["train"]["learning_rate"],
                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = build_scheduler(optim, total_steps, cfg["train"]["warmup_ratio"])

    n_train = sum(p.numel() for p in vlm.trainable_parameters())
    print(f"[info] trainable params: {n_train/1e6:.2f} M (projector only)")
    print(f"[info] steps / epoch: {steps_per_epoch} · total steps: {total_steps}")

    output_dir = Path(cfg["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm.train()
    global_step = 0
    t0 = time.time()
    running = 0.0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        optim.zero_grad()
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = vlm(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
                num_steps=cfg["train"]["num_steps_for_training"],
            )
            loss = out.loss / grad_accum
            loss.backward()
            running += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(vlm.trainable_parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1

                if global_step % cfg["train"]["logging_steps"] == 0:
                    avg = running / cfg["train"]["logging_steps"] / grad_accum
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}",
                                     step=global_step, sec=f"{elapsed:.0f}")
                    running = 0.0

                if global_step % cfg["train"]["save_steps"] == 0:
                    ckpt = output_dir / f"projector_step{global_step}.pt"
                    vlm.save_projector(str(ckpt))

    final = output_dir / "projector_final.pt"
    vlm.save_projector(str(final))
    print(f"[info] saved final projector -> {final}")


if __name__ == "__main__":
    main()
