"""LLaVA-1.5 pretrain data loader.

Expected layout (downloaded via scripts/download_llava.sh):
    data/llava/
        llava_pretrain_558k.json   # conversations file
        images/                    # flat image dir referenced by `image` field

Each sample's conversations alternate human / gpt turns. For pretrain alignment
we keep it very simple: one image + one caption-style response.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

IGNORE_INDEX = -100


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
        pixel_values = self.image_processor(images=img, return_tensors="pt")["pixel_values"][0]
        return pixel_values

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        # LLaVA pretrain format: {"image": "xxx.jpg", "conversations": [{from, value}, ...]}
        image_rel = sample["image"]
        convs = sample["conversations"]

        # First human turn contains the <image> placeholder; first gpt turn is target.
        human_text = convs[0]["value"].replace("<image>", self.image_token)
        # Ensure exactly one <image> token in human_text.
        if self.image_token not in human_text:
            human_text = self.image_token + "\n" + human_text
        target_text = convs[1]["value"].strip()

        # Build prompt with chat-like format. Keep it simple: Q ... A ...
        prompt_text = f"USER: {human_text}\nASSISTANT: "
        full_text = prompt_text + target_text + "\n"

        # Tokenize prompt and full sequence separately to get label mask.
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[len(prompt_ids):] = input_ids[len(prompt_ids):]

        pixel_values = self._load_image(image_rel)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
        }


def _expand_labels_for_image(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    image_token_id: int,
    num_image_patches: int,
) -> (torch.Tensor, torch.Tensor):
    """Given a 1D input_ids/labels pair that contains a single <image> token,
    expand that slot to `num_image_patches` positions, setting label = IGNORE
    for those positions. Returns the aligned (expanded_ids, expanded_labels).

    The expanded_ids are returned only to get correct lengths for attention masks
    — the actual embedding substitution happens in LoopedVLM.build_inputs_embeds
    using the *original* 1D input_ids, NOT the expanded one. But we still need
    lengths to line up for `labels`. So we construct expanded_labels of the
    final length, and we will use them with inputs_embeds from build_inputs_embeds.
    """
    mask = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if mask.numel() == 0:
        return input_ids, labels
    img_pos = mask[0].item()
    pre_l = labels[:img_pos]
    post_l = labels[img_pos + 1:]
    fill = torch.full((num_image_patches,), IGNORE_INDEX, dtype=labels.dtype)
    expanded_labels = torch.cat([pre_l, fill, post_l], dim=0)
    # expanded_ids is not used directly for embeddings, but is useful for debug.
    pre_i = input_ids[:img_pos]
    post_i = input_ids[img_pos + 1:]
    fill_i = torch.full((num_image_patches,), image_token_id, dtype=input_ids.dtype)
    expanded_ids = torch.cat([pre_i, fill_i, post_i], dim=0)
    return expanded_ids, expanded_labels


def collate_llava(batch: List[Dict], image_token_id: int, num_image_patches: int,
                  pad_token_id: int = 65509) -> Dict:
    """Pad, align labels to the inputs_embeds length produced by LoopedVLM."""
    # Expand labels to match the length AFTER image token substitution.
    expanded_input_ids = []
    expanded_labels = []
    for b in batch:
        eids, elab = _expand_labels_for_image(
            b["input_ids"], b["labels"], image_token_id, num_image_patches)
        expanded_input_ids.append(eids)
        expanded_labels.append(elab)

    max_len = max(x.size(0) for x in expanded_input_ids)

    def _pad_ids(t, pad_val):
        if t.size(0) == max_len:
            return t
        pad = torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    padded_input_ids = torch.stack([_pad_ids(x, pad_token_id)
                                    for x in expanded_input_ids])
    padded_labels = torch.stack([_pad_ids(x, IGNORE_INDEX)
                                 for x in expanded_labels])
    attention_mask = (padded_input_ids != pad_token_id).long()

    # We also need to pass the ORIGINAL (unexpanded) input_ids to the model so
    # it can do the embedding substitution. Pad those too.
    orig_max = max(b["input_ids"].size(0) for b in batch)
    orig_ids = torch.stack([
        _pad_ids(b["input_ids"], pad_token_id) if b["input_ids"].size(0) < orig_max
        else b["input_ids"] for b in batch
    ])
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    return {
        "input_ids": orig_ids,              # goes into LoopedVLM.forward
        "labels": padded_labels,            # pre-aligned to inputs_embeds length
        "attention_mask": attention_mask,   # aligned to inputs_embeds length
        "pixel_values": pixel_values,
    }
