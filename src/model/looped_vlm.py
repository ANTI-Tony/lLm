"""
LoopedVLM: CLIP vision encoder + MLP projector + frozen recurrent-depth LLM.

Design (after smoke test):
    * The LLM is FROZEN. We only train the projector. Finetuning the LLM would
      destroy the pretrained loop dynamics and invalidate H1.
    * CLIP is FROZEN. Standard LLaVA-1.5 recipe.
    * num_steps is threaded through forward / generate at inference.

Huginn-specific quirk:
    Huginn's forward signature is
        forward(input_ids, input_embeds=None, ..., num_steps=None, ...)
    Note `input_embeds` is singular (non-standard — HF usually uses
    `inputs_embeds`). We build substituted embeddings ourselves and pass BOTH
    `input_ids` (required positional) and `input_embeds` (our override).

Token alignment strategy:
    Each sample contains a single <image> marker in the raw text. The data
    pipeline expands that one token to `num_image_patches` consecutive
    image_token_id positions. At forward time we:
        1. Call the embedding layer on input_ids to get base embeddings.
        2. Replace the `num_image_patches` image_token_id positions with
           our CLIP-projected vision features.
        3. Pass input_ids + input_embeds + attention_mask to the LLM.
    This guarantees shape consistency (input_ids and input_embeds have the
    same sequence length) so Huginn's position-id / cache bookkeeping stays
    valid.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

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
        if cfg.image_placeholder not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([cfg.image_placeholder], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            cfg.image_placeholder)

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.vision = CLIPVisionModel.from_pretrained(cfg.vision_encoder)
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_encoder)

        vision_hidden = self.vision.config.hidden_size
        llm_hidden = self.llm.get_input_embeddings().weight.shape[1]
        self.projector = build_projector(cfg.projector_type, vision_hidden, llm_hidden)
        self.projector.to(torch_dtype)

        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
        if cfg.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

    # ---------- vision encoding ----------

    @torch.no_grad()
    def encode_image_frozen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision(pixel_values=pixel_values, output_hidden_states=False)
        return outputs.last_hidden_state[:, 1:, :]  # drop [CLS]

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image_frozen(pixel_values)
        return self.projector(feats)  # [B, num_patches, llm_hidden]

    # ---------- embedding substitution ----------

    def _build_input_embeds(
        self,
        input_ids: torch.Tensor,
        projected_vision: torch.Tensor,
    ) -> torch.Tensor:
        """input_ids must already contain exactly `num_patches` consecutive
        image_token_id positions per sample (data pipeline does this).
        Returns input_embeds with those positions replaced by projected_vision.
        """
        embed_layer = self.llm.get_input_embeddings()
        base = embed_layer(input_ids)  # [B, T, H]
        B, T, H = base.shape
        num_patches = projected_vision.size(1)

        for b in range(B):
            positions = (input_ids[b] == self.image_token_id
                         ).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            if positions.numel() != num_patches:
                raise ValueError(
                    f"sample {b} has {positions.numel()} image tokens but "
                    f"vision has {num_patches} patches. The data pipeline must "
                    f"insert exactly num_patches image tokens.")
            start = positions[0].item()
            end = positions[-1].item() + 1
            if end - start != num_patches:
                raise ValueError(
                    f"image tokens at sample {b} are not contiguous "
                    f"(start={start}, end={end}, num_patches={num_patches}).")
            base[b, start:end, :] = projected_vision[b]
        return base

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
            input_embeds = self._build_input_embeds(input_ids, projected)
            return self.llm(
                input_ids=input_ids,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                num_steps=num_steps,
            )
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            num_steps=num_steps,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        max_new_tokens: int = 64,
        gen_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        """For the vision path we run a manual greedy loop because the
        HuggingFace generate() pipeline does not thread `input_embeds` per
        step. Text-only path still uses model.generate() for speed.
        """
        if gen_config is None:
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None, top_k=None, top_p=None, min_p=None,
                return_dict_in_generate=True,
                eos_token_id=65505, bos_token_id=65504, pad_token_id=65509,
                use_cache=True,
            )

        if pixel_values is None:
            out = self.llm.generate(
                input_ids,
                tokenizer=self.tokenizer,
                num_steps=num_steps,
                generation_config=gen_config,
            )
            return out.sequences if hasattr(out, "sequences") else out

        # ----- vision path: manual greedy decode -----
        device = input_ids.device
        eos_id = int(gen_config.eos_token_id) if gen_config.eos_token_id is not None else 65505
        projected = self.encode_image(pixel_values)
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            input_embeds = self._build_input_embeds(generated, projected)
            out = self.llm(
                input_ids=generated,
                input_embeds=input_embeds,
                num_steps=num_steps,
            )
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return generated

    # ---------- utilities ----------

    def trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]

    def save_projector(self, path: str):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(sd)
