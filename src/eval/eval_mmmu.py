"""MMMU evaluation sweeping num_steps.

Dataset: lmms-lab/MMMU (HuggingFace). Validation set ~900 questions.
MMMU is multiple-choice; we score by letter-exact match.

Usage:
    python -m src.eval.eval_mmmu --config configs/huginn_vlm.yaml \
        --projector checkpoints/projector_huginn/projector_final.pt
"""

import argparse
from typing import Dict

import yaml
from datasets import load_dataset

from src.eval.eval_common import load_vlm, sweep_benchmark
from src.utils.answer_parse import extract_mcq_letter


def _to_sample(row) -> Dict | None:
    # MMMU fields: id, question, options (list of strings), answer (letter),
    # image_1, image_2, ...
    if row.get("image_1") is None:
        return None
    choices = row.get("options")
    if isinstance(choices, str):
        import ast
        try:
            choices = ast.literal_eval(choices)
        except Exception:
            choices = [choices]
    gold = row.get("answer")
    if not gold:
        return None
    return {
        "id": row.get("id"),
        "question": row["question"],
        "choices": choices,
        "gold": gold.strip().upper(),
        "image": row["image_1"].convert("RGB"),
    }


def score(sample, pred: str) -> bool:
    letter = extract_mcq_letter(pred, sample.get("choices"))
    return letter is not None and letter == sample["gold"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--projector", default=None,
                        help="projector checkpoint (.pt). Skip for zero-shot "
                             "(untrained projector) sanity check.")
    parser.add_argument("--output", default="results/mmmu.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    bench = cfg["eval"]["benchmarks"]["mmmu"]
    max_samples = args.max_samples or bench["max_samples"]

    vlm = load_vlm(args.config, projector_ckpt=args.projector)

    print("[info] loading MMMU ...")
    # lmms-lab/MMMU's current HF card only exposes the "default" config
    # (the "Overall" subset name was deprecated). Pass no name to pick up
    # the default.
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
