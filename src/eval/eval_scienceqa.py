"""ScienceQA evaluation sweeping num_steps.

Dataset: derek-thomas/ScienceQA. We use the subset that has an image.
"""

import argparse
from typing import Dict

import yaml
from datasets import load_dataset

from src.eval.eval_common import load_vlm, sweep_benchmark
from src.utils.answer_parse import extract_mcq_letter


def _to_sample(row) -> Dict | None:
    img = row.get("image")
    if img is None:
        return None
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    choices = row.get("choices")
    if not choices:
        return None
    gold_idx = row.get("answer")
    if gold_idx is None:
        return None
    gold_letter = chr(ord("A") + int(gold_idx))
    q = row.get("question")
    hint = row.get("hint") or ""
    if hint:
        q = q + "\nHint: " + hint
    return {
        "id": row.get("id") or row.get("image_id"),
        "question": q,
        "choices": choices,
        "gold": gold_letter,
        "image": img,
    }


def score(sample, pred: str) -> bool:
    letter = extract_mcq_letter(pred, sample.get("choices"))
    return letter is not None and letter == sample["gold"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--projector", default=None)
    parser.add_argument("--output", default="results/scienceqa.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    bench = cfg["eval"]["benchmarks"]["scienceqa"]
    max_samples = args.max_samples or bench["max_samples"]

    vlm = load_vlm(args.config, projector_ckpt=args.projector)

    print("[info] loading ScienceQA ...")
    ds = load_dataset(bench["dataset"], split=bench["split"])

    def iterator():
        n = 0
        for row in ds:
            s = _to_sample(row)
            if s is None:
                continue
            yield s
            n += 1
            if max_samples and n >= max_samples:
                break

    sweep_benchmark(
        vlm,
        iterator=iterator(),
        scorer=score,
        num_steps_list=cfg["eval"]["num_steps_sweep"],
        image_token=vlm.cfg.image_placeholder,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        total=max_samples,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
