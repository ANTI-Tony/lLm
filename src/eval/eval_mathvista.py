"""MathVista evaluation sweeping num_steps.

Dataset: lmms-lab/MathVista. Split: testmini (~1000 samples).
MathVista has both MCQ and free-form numeric answers.
"""

import argparse
from typing import Dict

import yaml
from datasets import load_dataset

from src.eval.eval_common import load_vlm, sweep_benchmark
from src.utils.answer_parse import extract_mcq_letter, numeric_extract, norm_text


def _to_sample(row) -> Dict | None:
    if row.get("decoded_image") is None and row.get("image") is None:
        return None
    img = row.get("decoded_image") or row.get("image")
    if hasattr(img, "convert"):
        img = img.convert("RGB")
    choices = row.get("choices")
    answer = row.get("answer")
    if answer is None:
        return None
    q = row.get("query") or row.get("question")
    return {
        "id": row.get("pid") or row.get("id"),
        "question": q,
        "choices": choices,
        "gold": str(answer).strip(),
        "question_type": row.get("question_type", "free_form"),
        "image": img,
    }


def score(sample, pred: str) -> bool:
    if sample.get("question_type") == "multi_choice" or sample.get("choices"):
        letter = extract_mcq_letter(pred, sample.get("choices"))
        if letter is None:
            return False
        # gold may be letter or option text.
        gold = sample["gold"].strip()
        if len(gold) == 1 and gold.upper() == letter:
            return True
        if sample.get("choices"):
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(sample["choices"]):
                return norm_text(sample["choices"][idx]) == norm_text(gold)
        return False
    # free-form -> extract a number, compare loosely.
    gold_num = numeric_extract(sample["gold"])
    pred_num = numeric_extract(pred)
    if gold_num is not None and pred_num is not None:
        try:
            return abs(float(gold_num) - float(pred_num)) < 1e-3
        except ValueError:
            return False
    return norm_text(pred) == norm_text(sample["gold"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--projector", default=None)
    parser.add_argument("--output", default="results/mathvista.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    bench = cfg["eval"]["benchmarks"]["mathvista"]
    max_samples = args.max_samples or bench["max_samples"]

    vlm = load_vlm(args.config, projector_ckpt=args.projector)

    print("[info] loading MathVista ...")
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
