"""GSM8K loader that works inside the GFW.

HF datasets is unreachable from the lab server (huggingface.co SSL fails
even via clashon proxy). ModelScope's AI-ModelScope/gsm8k mirror works
without proxy. This helper hides the indirection — callers just want a
list of {question, answer} dicts.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=4)
def load_gsm8k(split: str = "test"):
    """Returns a list of dicts with 'question' and 'answer' keys.

    Tries Hugging Face first (in case proxy is reliable); falls back to
    ModelScope mirror (preferred default in this project).
    """
    # In-China lab: HF / hf-mirror unreachable. Skip HF, go straight to MS.
    # If you need HF, set USE_HF_GSM8K=1 to override.
    if os.environ.get("USE_HF_GSM8K") == "1":
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=split)
            print(f"[data_utils] gsm8k via HF: {len(ds)} samples")
            return [{"question": x["question"], "answer": x["answer"]} for x in ds]
        except Exception as e_hf:
            pass

    from modelscope import MsDataset
    ms_split = "test" if split == "test" else "train"
    ds = MsDataset.load("AI-ModelScope/gsm8k",
                        subset_name="main",
                        split=ms_split)
    out = [{"question": x["question"], "answer": x["answer"]} for x in ds]
    print(f"[data_utils] gsm8k via ModelScope: {len(out)} samples ({split})")
    return out
