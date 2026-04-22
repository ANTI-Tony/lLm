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

# cuDNN 9.19 shipped in torch 2.5.1+cu121 is broken on this RunPod image
# (CUDNN_STATUS_NOT_INITIALIZED). Disable it here so any script that imports
# LoopedVLM is protected, not just the training/eval entry points.
torch.backends.cudnn.enabled = False
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
    # H1 requires the recurrent core to stay untouched. lm_head (output
    # projection) and ln_f (final layer norm) are NOT part of the loop —
    # they run once after the recurrent block. Unfreezing them lets the
    # LLM adapt to the projector's output distribution without changing
    # loop dynamics, which is essential for making a base (non-instruct)
    # LLM like Huginn work in a frozen-LLM VLM setup. v1/v2 both
    # collapsed with everything frozen.
    unfreeze_coda: bool = True
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
        base = self.original(input_ids)  # frozen -> no grad
        if self._vision_features is None:
            return base

        # Three regimes to distinguish:
        #   (a) positions == 0    — no image in this batch at all. Normal
        #                           during KV-cached generation of new tokens.
        #                           Return base unmodified.
        #   (b) 0 < positions < N — partial image block, indicates truncation
        #                           or data-pipeline bug. Hard fail.
        #   (c) positions >= N    — full image block, substitute.
        B = input_ids.size(0)
        num_patches = self._vision_features.size(1)
        out_rows = []
        truncated = []
        for b in range(B):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            npos = positions.numel()
            if npos == 0:
                out_rows.append(base[b])
                continue
            if npos < num_patches:
                truncated.append((b, npos))
                out_rows.append(base[b])
                continue
            start = positions[0].item()
            end = start + num_patches
            row = torch.cat(
                [base[b, :start],
                 self._vision_features[b].to(base.dtype),
                 base[b, end:]],
                dim=0,
            )
            out_rows.append(row)
        if truncated:
            raise RuntimeError(
                f"VisionAwareEmbedding: {len(truncated)} batch elements had "
                f"0 < image_tokens < {num_patches} (details: {truncated[:4]}). "
                "The data pipeline must insert exactly num_image_patches "
                "image tokens per sample; max_seq_length is likely truncating "
                "the image block.")
        return torch.stack(out_rows, dim=0)

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
        # image_token_id is set after we expand the model's embedding below.

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Huginn doesn't implement set_input_embeddings, and its _init_weights
        # keys on an internal module-id lookup. Both combine to make HF's
        # resize_token_embeddings fail. Expand the embedding manually instead.
        # We deliberately do NOT touch the lm_head: the new token id is only
        # ever used as an input marker, never produced as output, so leaving
        # the output vocab at 65536 is correct and avoids tied-weight headaches.
        self.image_token_id = self._expand_input_embedding()
        # Sync tokenizer so its string -> id mapping matches what we did.
        self.tokenizer.add_tokens([cfg.image_placeholder], special_tokens=True)
        tok_assigned = self.tokenizer.convert_tokens_to_ids(cfg.image_placeholder)
        if tok_assigned != self.image_token_id:
            # Rare case: tokenizer already had added tokens. Just use the
            # tokenizer's id and write a second row there too.
            raise RuntimeError(
                f"tokenizer assigned <image> id {tok_assigned} but we expanded "
                f"embedding to row {self.image_token_id}. Inspect the tokenizer.")

        # Install the vision-aware embedding hook. Huginn doesn't implement
        # set_input_embeddings, so we locate the embedding attribute manually
        # and swap it in place.
        original_embed = self.llm.get_input_embeddings()
        self._vision_embed = VisionAwareEmbedding(original_embed, self.image_token_id)
        self._replace_submodule(original_embed, self._vision_embed)

        # Load CLIP in the same dtype as the LLM. Loading in fp32 (the HF
        # default) then mixing with bf16 activations downstream triggers
        # cuDNN algorithm-selection issues on some CUDA 12.1 / driver 570
        # combos; matching dtypes is the cleanest fix.
        self.vision = CLIPVisionModel.from_pretrained(
            cfg.vision_encoder, torch_dtype=torch_dtype)
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

        # Unfreeze the coda (non-recurrent output-side params). These run
        # ONCE after the recurrent loop, so unfreezing them does not change
        # loop dynamics. Concretely: self.llm.lm_head and
        # self.llm.transformer.ln_f. We also unfreeze the newly-added
        # image-token row of the input embedding, though since the hook
        # overwrites that row's output it shouldn't matter.
        if cfg.unfreeze_coda:
            unfrozen = []
            try:
                for p in self.llm.lm_head.parameters():
                    p.requires_grad = True
                unfrozen.append("lm_head")
            except AttributeError:
                pass
            try:
                for p in self.llm.transformer.ln_f.parameters():
                    p.requires_grad = True
                unfrozen.append("transformer.ln_f")
            except AttributeError:
                pass
            print(f"[info] unfroze coda params: {unfrozen}")

    # ---------- manual embedding surgery (bypasses HF resize / set_input) ----

    def _replace_submodule(self, old: nn.Module, new: nn.Module) -> None:
        """Find `old` in the LLM's module tree and replace it with `new`."""
        for name, mod in self.llm.named_modules():
            if mod is old:
                parent_name, _, attr = name.rpartition('.')
                parent = (self.llm if not parent_name
                          else self.llm.get_submodule(parent_name))
                setattr(parent, attr, new)
                return
        raise RuntimeError("could not locate target submodule in LLM")

    def _expand_input_embedding(self) -> int:
        """Append one row to the LLM's input embedding matrix. Returns the
        new token's id (= old_vocab_size)."""
        old = self.llm.get_input_embeddings()
        old_num, hidden = old.weight.shape
        dtype, device = old.weight.dtype, old.weight.device

        new_emb = nn.Embedding(old_num + 1, hidden).to(dtype).to(device)
        with torch.no_grad():
            new_emb.weight[:old_num].copy_(old.weight)
            nn.init.normal_(new_emb.weight[old_num:], mean=0.0, std=0.02)

        self._replace_submodule(old, new_emb)
        return old_num

    # ---------- vision encoding ----------

    @torch.no_grad()
    def encode_image_frozen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Cast pixel_values to the vision encoder's dtype to avoid fp32/bf16
        # mix that upsets cuDNN algorithm selection.
        pv_dtype = next(self.vision.parameters()).dtype
        outputs = self.vision(pixel_values=pixel_values.to(pv_dtype),
                              output_hidden_states=False)
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
        # Huginn's iterate_forward splits num_steps into
        # (num_steps_no_grad, num_steps_with_grad). With a plain int it
        # dispatches (num_steps, 0) — i.e. every iteration runs inside
        # torch.no_grad(), which detaches the whole forward from the
        # autograd graph and makes projector training impossible.
        # Passing a tuple (0, N) forces all N iterations to run with grad.
        if isinstance(num_steps, int):
            huginn_num_steps = (0, num_steps)
        else:
            huginn_num_steps = num_steps

        if pixel_values is None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                num_steps=huginn_num_steps,
            )

        # Build input_embeds via the hook so the projector is on the
        # autograd graph, then pass explicitly to Huginn.
        projected = self.encode_image(pixel_values)
        self._set_vision(projected)
        try:
            input_embeds = self._vision_embed(input_ids)
            return self.llm(
                input_ids=input_ids,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                labels=labels,
                num_steps=huginn_num_steps,
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

        # At inference we want all iterations in no_grad for speed, which is
        # exactly what Huginn does when given an int. No tuple conversion.
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
        params = list(self.projector.parameters())
        # Include unfrozen LLM coda params (lm_head, ln_f) if any.
        for p in self.llm.parameters():
            if p.requires_grad:
                params.append(p)
        return [p for p in params if p.requires_grad]

    def save_projector(self, path: str):
        """Save projector + any unfrozen coda params so eval can restore them."""
        state = {"projector": self.projector.state_dict()}
        coda = {}
        for p_name, p in self.llm.named_parameters():
            if p.requires_grad:
                coda[p_name] = p.detach().cpu()
        if coda:
            state["coda"] = coda
        torch.save(state, path)

    def load_projector(self, path: str):
        sd = torch.load(path, map_location="cpu")
        # Backwards compat: v1/v2 checkpoints saved just the projector dict.
        if "projector" not in sd:
            self.projector.load_state_dict(sd)
            return
        self.projector.load_state_dict(sd["projector"])
        if "coda" in sd:
            llm_sd = dict(self.llm.named_parameters())
            for k, v in sd["coda"].items():
                if k in llm_sd:
                    llm_sd[k].data.copy_(v.to(llm_sd[k].device,
                                              dtype=llm_sd[k].dtype))
