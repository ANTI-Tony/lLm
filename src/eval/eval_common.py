"""Shared evaluation helpers: build VLM, iterate benchmark, sweep num_steps."""

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import torch
import yaml
from tqdm import tqdm

# Same cuDNN issue as training: broken cuDNN on this RunPod cu121 image.
torch.backends.cudnn.enabled = False

from src.model.looped_vlm import LoopedVLM, LoopedVLMConfig


def load_vlm(cfg_path: str, projector_ckpt: str | None = None) -> LoopedVLM:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = LoopedVLMConfig(
        llm_name=cfg["model"]["llm_name"],
        vision_encoder=cfg["model"]["vision_encoder"],
        projector_type=cfg["model"]["projector_type"],
        freeze_llm=True,
        freeze_vision=True,
    )
    vlm = LoopedVLM(model_cfg, torch_dtype=torch.bfloat16).to(device).eval()
    if projector_ckpt:
        vlm.load_projector(projector_ckpt)
        print(f"[info] loaded projector from {projector_ckpt}")
    return vlm


def build_prompt(question: str, choices: List[str] | None, image_token: str) -> str:
    img = image_token + "\n"
    if choices:
        opts = "\n".join(f"{chr(ord('A')+i)}. {c}" for i, c in enumerate(choices))
        return (f"USER: {img}{question}\n{opts}\n"
                f"Answer with a single letter.\nASSISTANT: ")
    return f"USER: {img}{question}\nASSISTANT: "


@torch.no_grad()
def generate_one(vlm: LoopedVLM, prompt: str, pil_image, num_steps: int,
                 max_new_tokens: int = 64) -> str:
    device = next(vlm.parameters()).device
    px = vlm.image_processor(images=pil_image, return_tensors="pt"
                             )["pixel_values"].to(device, dtype=torch.bfloat16)
    ids = vlm.tokenizer.encode(prompt, return_tensors="pt",
                               add_special_tokens=True).to(device)
    seq = vlm.generate(input_ids=ids, pixel_values=px,
                       num_steps=num_steps, max_new_tokens=max_new_tokens)
    # strip the prompt tokens to get only the generated portion.
    # `seq` from generate(inputs_embeds=...) returns only new tokens in many HF
    # versions; be defensive and decode whatever comes back.
    text = vlm.tokenizer.decode(seq[0], skip_special_tokens=True)
    # If the prompt appears at the start, cut it out.
    if text.startswith(prompt):
        text = text[len(prompt):]
    # Heuristic: trim at USER: to avoid multi-turn echo.
    text = text.split("USER:")[0].strip()
    return text


def sweep_benchmark(
    vlm: LoopedVLM,
    iterator: Iterable[Dict],
    scorer: Callable[[Dict, str], bool],
    num_steps_list: List[int],
    image_token: str,
    max_new_tokens: int,
    total: int | None,
    save_path: str,
) -> Dict:
    """For each sample, generate at every num_steps value. Record per-sample
    per-num_steps prediction + correctness. Save as JSON.
    """
    per_sample = []
    counts = {n: [0, 0] for n in num_steps_list}  # correct, total
    t0 = time.time()
    for i, sample in enumerate(tqdm(iterator, total=total)):
        prompt = build_prompt(sample["question"],
                              sample.get("choices"),
                              image_token=image_token)
        entry = {"id": sample.get("id", i),
                 "question": sample["question"],
                 "choices": sample.get("choices"),
                 "gold": sample.get("gold"),
                 "preds": {}}
        for n in num_steps_list:
            pred = generate_one(vlm, prompt, sample["image"], num_steps=n,
                                max_new_tokens=max_new_tokens)
            ok = scorer(sample, pred)
            entry["preds"][str(n)] = {"text": pred, "correct": bool(ok)}
            counts[n][0] += int(ok)
            counts[n][1] += 1
        per_sample.append(entry)

    summary = {"num_steps_accuracy": {str(n): (c[0] / c[1] if c[1] else 0.0)
                                      for n, c in counts.items()},
               "num_samples": total,
               "elapsed_seconds": time.time() - t0}
    out = {"summary": summary, "per_sample": per_sample}

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[info] saved -> {save_path}")
    print(json.dumps(summary, indent=2))
    return out
