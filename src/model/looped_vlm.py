"""
LoopedVLM: CLIP vision encoder + MLP projector + frozen recurrent-depth LLM.

Design:
    * LLM + CLIP frozen. Only the projector trains. Finetuning the LLM would
      destroy Huginn's pretrained loop dynamics and invalidate H1.
    * Vision injection via embedding-layer hook: we wrap Huginn's input
      embedding layer with VisionAwareEmbedding. When input_ids contains
      num_patches consecutive image_token_id positions and we have set
      `_vision_features`, those positions' embeddings are replaced.
    * With the hook in place, both training (forward with labels) and
      inference (model.generate()) work through Huginn's normal code paths.
      This means HF's KV cache machinery kicks in automatically for
      generation — essential for eval throughput.

Why a hook instead of passing input_embeds directly:
    Huginn's forward accepts input_embeds (singular), but HF's generate()
    pipeline does not thread that kwarg per-step. Passing input_embeds
    manually forces us to write a cache-free O(T^2) greedy loop, which is
    25x too slow for Phase 0 eval.
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


class VisionAwareEmbedding(nn.Module):
    """Drop-in replacement for an nn.Embedding layer.

    Forward behaviour:
        * Always calls the original embedding on input_ids.
        * If `_vision_features` is set (shape [B, num_patches, H]) and a
          sample in the batch has >= num_patches consecutive image_token_id
          positions, the first num_patches of those positions are overwritten
          with the projected vision features.
        * During cached generation, input_ids for new tokens contain no
          image_token_id, so the hook is a no-op — standard behaviour.
    """

    def __init__(self, original: nn.Embedding, image_token_id: int):
        super().__init__()
        self.original = original
        self.image_token_id = image_token_id
        self._vision_features: Optional[torch.Tensor] = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.original(input_ids)
        if self._vision_features is None:
            return embeds

        B = input_ids.size(0)
        num_patches = self._vision_features.size(1)
        for b in range(B):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() < num_patches:
                continue
            start = positions[0].item()
            end = start + num_patches
            embeds[b, start:end] = self._vision_features[b].to(embeds.dtype)
        return embeds

    # Proxy common Embedding attributes so resize/tie-weights still work.
    @property
    def weight(self):
        return self.original.weight

    @property
    def num_embeddings(self):
        return self.original.num_embeddings

    @property
    def embedding_dim(self):
        return self.original.embedding_dim


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
        # mean_resizing=False skips the multivariate-normal init that HF does
        # by default. For Huginn (vocab=65536, hidden=5280) that Cholesky takes
        # 1-2 min and has no value here: the new row for <image> is never read
        # at forward time — the VisionAwareEmbedding hook overwrites it with
        # projected vision features.
        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        # Install the vision-aware embedding hook.
        original_embed = self.llm.get_input_embeddings()
        self._vision_embed = VisionAwareEmbedding(original_embed, self.image_token_id)
        self.llm.set_input_embeddings(self._vision_embed)

        self.vision = CLIPVisionModel.from_pretrained(cfg.vision_encoder)
        self.image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_encoder)

        vision_hidden = self.vision.config.hidden_size
        llm_hidden = self._vision_embed.original.weight.shape[1]
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

    def _set_vision(self, projected: Optional[torch.Tensor]):
        self._vision_embed._vision_features = projected

    # ---------- forward / generate ----------

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_steps: int = 16,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        projected = None
        if pixel_values is not None:
            projected = self.encode_image(pixel_values)
            self._set_vision(projected)
        try:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_steps=num_steps,
            )
        finally:
            self._set_vision(None)

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

        projected = None
        if pixel_values is not None:
            projected = self.encode_image(pixel_values)
            self._set_vision(projected)
        try:
            out = self.llm.generate(
                input_ids,
                tokenizer=self.tokenizer,
                num_steps=num_steps,
                generation_config=gen_config,
            )
        finally:
            self._set_vision(None)
        return out.sequences if hasattr(out, "sequences") else out

    # ---------- utilities ----------

    def trainable_parameters(self):
        return [p for p in self.projector.parameters() if p.requires_grad]

    def save_projector(self, path: str):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path: str):
        sd = torch.load(path, map_location="cpu")
        self.projector.load_state_dict(sd)
