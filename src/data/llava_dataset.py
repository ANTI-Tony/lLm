"""LLaVA-1.5 pretrain data loader.

Expected layout (downloaded via scripts/download_llava.sh):
    data/llava/
        llava_pretrain_558k.json   # conversations file
        images/                    # image subdirs referenced by `image` field

Key design: the raw <image> placeholder token is expanded *at __getitem__
time* to exactly `num_image_patches` consecutive image_token_id positions.
This keeps input_ids and input_embeds the same length, which Huginn needs
for position-id / cache bookkeeping.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

IGNORE_INDEX = -100


def _expand_image_tokens(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    image_token_id: int,
    num_image_patches: int,
) -> (torch.Tensor, torch.Tensor):
    """Expand each <image> token (id == image_token_id) into num_image_patches
    consecutive image tokens. Labels at the expanded positions are set to
    IGNORE_INDEX. Only supports one image per sample (asserted).
    """
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return input_ids, labels
    assert positions.numel() == 1, \
        f"expected exactly one <image> token, got {positions.numel()}"
    img_pos = positions[0].item()

    pre_ids, post_ids = input_ids[:img_pos], input_ids[img_pos + 1:]
    fill_ids = torch.full((num_image_patches,), image_token_id,
                          dtype=input_ids.dtype)
    new_ids = torch.cat([pre_ids, fill_ids, post_ids], dim=0)

    pre_lbl, post_lbl = labels[:img_pos], labels[img_pos + 1:]
    fill_lbl = torch.full((num_image_patches,), IGNORE_INDEX,
                          dtype=labels.dtype)
    new_lbl = torch.cat([pre_lbl, fill_lbl, post_lbl], dim=0)
    return new_ids, new_lbl


class LlavaPretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer,
        image_processor,
        image_token: str,
        image_token_id: int,
        num_image_patches: int,
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        with open(data_path, "r") as f:
            raw = json.load(f)
        if max_samples is not None:
            raw = raw[:max_samples]
        self.data = raw
        self.image_folder = Path(image_folder)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.num_image_patches = num_image_patches
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        path = self.image_folder / rel_path
        img = Image.open(path).convert("RGB")
        pixel_values = self.image_processor(images=img, return_tensors="pt"
                                            )["pixel_values"][0]
        return pixel_values

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        image_rel = sample["image"]
        convs = sample["conversations"]

        human_text = convs[0]["value"].replace("<image>", self.image_token)
        if self.image_token not in human_text:
            human_text = self.image_token + "\n" + human_text
        target_text = convs[1]["value"].strip()

        prompt_text = f"USER: {human_text}\nASSISTANT: "
        full_text = prompt_text + target_text + "\n"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[len(prompt_ids):] = input_ids[len(prompt_ids):]

        # Expand <image> -> N copies NOW so downstream shapes all line up.
        input_ids, labels = _expand_image_tokens(
            input_ids, labels, self.image_token_id, self.num_image_patches)

        # Truncate if needed (keep the image block — cut from the tail).
        if input_ids.size(0) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        pixel_values = self._load_image(image_rel)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def collate_llava(batch: List[Dict], pad_token_id: int = 65509) -> Dict:
    """Pad all tensors to the batch max length. input_ids and labels now have
    matching shape because __getitem__ already expanded <image>."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    def _pad(t, pad_val):
        if t.size(0) == max_len:
            return t
        pad = torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    input_ids = torch.stack([_pad(b["input_ids"], pad_token_id) for b in batch])
    labels = torch.stack([_pad(b["labels"], IGNORE_INDEX) for b in batch])
    attention_mask = (input_ids != pad_token_id).long()
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
