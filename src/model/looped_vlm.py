"""
LoopedVLM: CLIP vision encoder + MLP projector + frozen recurrent-depth LLM.

Design choices (Phase 0):
    * LLM is FROZEN. We only train the projector. This preserves the looped
      depth dynamics that Ouro/Huginn learned during pretraining. Finetuning
      the LLM would risk destroying those dynamics, invalidating H1.
    * CLIP is FROZEN. Standard LLaVA-1.5 pretrain recipe.
    * num_steps is threaded through forward / generate at inference. During
      alignment training we use a fixed moderate value (config.num_steps_for_training).

Adapter strategy:
    We use the classic LLaVA approach: insert a single `<image>` placeholder
    token in the prompt, then substitute its embedding with the projected
    vision features at forward time. No modeling_huginn surgery required.

IMPORTANT: if the smoke test shows inputs_embeds does not work, this class
needs to be rewritten. Read smoke_test.py check-2 before committing to this.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor,
    GenerationConfig,
)

from .projector import build_projector


IMAGE_PLACEHOLDER = "<image>"


@dataclass
class LoopedVLMConfig:
    llm_name: str
    vision_encoder: str
    projector_type: str = "mlp2x_gelu"
    freeze_llm: bool = True
    freeze_vision: bool = True
    image_placeholder: str = IMAGE_PLACEHOLDER


class LoopedVLM(nn.Module):
    def __init__(self, cfg: LoopedVLMConfig, torch_dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        # Reserve an id for <image>. Huginn's tokenizer does not have it.
        if cfg.image_placeholder not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([cfg.image_placeholder], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            cfg.image_placeholder)

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        # We added a token — resize embeddings (this *does* add a trainable row,
        # which is fine; we do not freeze the embedding layer rows we just added).
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.vision = CLIPVisionModel.from_pretrained(cfg.vision_encoder)
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_encoder)

        vision_hidden = self.vision.config.hidden_size
        llm_hidden = self.llm.get_input_embeddings().weight.shape[1]
        self.projector = build_projector(cfg.projector_type, vision_hidden, llm_hidden)

        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            # Keep the embedding row for <image> trainable? No — we always
            # replace that slot with projected features at forward time, so
            # the embedding row is never read. Leave it frozen.

        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.projector.to(torch_dtype)

    # ---------- vision encoding ----------

    @torch.no_grad()
    def encode_image_frozen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return CLIP patch features (no [CLS]), no grad."""
        outputs = self.vision(pixel_values=pixel_values, output_hidden_states=False)
        # drop [CLS]
        return outputs.last_hidden_state[:, 1:, :]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image_frozen(pixel_values)
        return self.projector(feats)  # [B, num_patches, llm_hidden]

    # ---------- text + vision -> inputs_embeds ----------

    def build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        projected_vision: torch.Tensor,
    ) -> torch.Tensor:
        """Replace every <image> token embedding with the projected vision seq.

        input_ids:        [B, T]   (may contain image placeholder tokens)
        projected_vision: [B, P, H] (already through projector)
        returns inputs_embeds: [B, T', H] with each image slot expanded to P tokens.
        """
        B = input_ids.size(0)
        embed_layer = self.llm.get_input_embeddings()
        token_embeds = embed_layer(input_ids)  # [B, T, H]

        out_rows = []
        for b in range(B):
            ids = input_ids[b]
            mask = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                out_rows.append(token_embeds[b])
                continue
            # For Phase 0 we support a single image per sample.
            img_pos = mask[0].item()
            pre = token_embeds[b, :img_pos]
            post = token_embeds[b, img_pos + 1:]
            merged = torch.cat([pre, projected_vision[b], post], dim=0)
            out_rows.append(merged)

        max_len = max(r.size(0) for r in out_rows)
        H = token_embeds.size(-1)
        padded = torch.zeros(B, max_len, H, dtype=token_embeds.dtype,
                             device=token_embeds.device)
        attn = torch.zeros(B, max_len, dtype=torch.long,
                           device=token_embeds.device)
        for b, r in enumerate(out_rows):
            padded[b, : r.size(0)] = r
            attn[b, : r.size(0)] = 1
        return padded, attn

    # ---------- forward / generate ----------

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if pixel_values is not None:
            projected = self.encode_image(pixel_values)
            inputs_embeds, built_attn = self.build_inputs_embeds(input_ids, projected)
            if labels is not None:
                # We need to pad labels to the new length with -100 where vision
                # tokens were inserted. The simplest correct thing: rebuild labels
                # aligned with inputs_embeds. See data.collate for how labels are
                # produced — they are already aligned with inputs_embeds length
                # because the collator inserts the expansion. For safety here we
                # only accept pre-aligned labels.
                if labels.size(1) != inputs_embeds.size(1):
                    raise ValueError(
                        f"labels length {labels.size(1)} != inputs_embeds length "
                        f"{inputs_embeds.size(1)}. The data collator must align them.")
            attention_mask = attention_mask if attention_mask is not None else built_attn
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                num_steps=num_steps,
            )
        # text-only path
        return self.llm(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        num_steps=num_steps)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        max_new_tokens: int = 64,
        gen_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        if gen_config is None:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_k=None, top_p=None, min_p=None,
                return_dict_in_generate=True,
                eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
                use_cache=True,
            )
        if pixel_values is not None:
            projected = self.encode_image(pixel_values)
            inputs_embeds, attn = self.build_inputs_embeds(input_ids, projected)
            out = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                tokenizer=self.tokenizer,
                num_steps=num_steps,
                generation_config=gen_config,
            )
        else:
            out = self.llm.generate(
                input_ids,
                tokenizer=self.tokenizer,
                num_steps=num_steps,
                generation_config=gen_config,
            )
        return out.sequences if hasattr(out, "sequences") else out

    # ---------- utilities ----------

    def trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]

    def save_projector(self, path: str):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(sd)
