"""
Looped Llama wrapper for A2 sanity check.

Idea: take a pretrained dense LLM, designate the last L_loop layers as a
"recurrent block" and apply them K times during forward. The base model
weights are unchanged; only the forward graph is modified. SFT will teach
those last layers to behave usefully under iteration.

This is the minimum-viable retrofit — no extra parameters, no architecture
surgery beyond running existing layers in a loop. If this can't beat
K=1 SFT on GSM8K after a small training run, the whole A2 idea is dead.

Usage from another script:
    from looped_llama import LoopedLlama
    model = LoopedLlama.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        n_loop_layers=4,
        K=4,
        input_injection=True,
        injection_scale=0.1,
    )
    out = model(input_ids=..., labels=...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class LoopedLlamaConfig:
    base_model: str = "meta-llama/Llama-3.2-1B"
    n_loop_layers: int = 4         # how many of the LAST layers form the loop block
    K: int = 4                     # number of times to apply the loop block
    input_injection: bool = True   # add scaled h_init each iteration (UT-style)
    injection_scale: float = 0.1   # scale factor for input injection
    loop_layernorm: bool = True    # add LN after each iteration to prevent blow-up


class LoopedLlama(nn.Module):
    """Drop-in wrapper around a HuggingFace LlamaForCausalLM.

    During forward, the last `n_loop_layers` decoder layers are applied K
    times. Everything else (embedding, earlier layers, norm, lm_head) is
    untouched. Loss is computed on labels using the post-loop hidden states.
    """

    def __init__(self, cfg: LoopedLlamaConfig, hf_model=None,
                 torch_dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        if hf_model is None:
            hf_model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model, torch_dtype=torch_dtype)
        self.llm = hf_model

        # Sanity check: layer count
        n_total = len(self.llm.model.layers)
        if cfg.n_loop_layers >= n_total:
            raise ValueError(
                f"n_loop_layers={cfg.n_loop_layers} must be < total layers {n_total}")
        self._n_static = n_total - cfg.n_loop_layers

        # Single shared LN across iterations (Universal Transformer style).
        # Avoids tying LN count to K so we can change K at forward time.
        if cfg.loop_layernorm:
            hidden = self.llm.config.hidden_size
            self.loop_ln = nn.LayerNorm(hidden, eps=1e-5).to(torch_dtype)

    @classmethod
    def from_pretrained(cls, base_model: str, **kwargs):
        cfg = LoopedLlamaConfig(base_model=base_model, **kwargs)
        return cls(cfg)

    # ----- forward -----

    def _build_attention_mask_4d(self, attention_mask, dtype, device):
        """Convert 2D attention mask to 4D causal mask expected by Llama layers."""
        bsz, seq_len = attention_mask.shape
        # Causal lower triangular
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool,
                                       device=device))
        # Combine with padding mask (False where padding).
        pad = attention_mask.bool()  # [B, S]
        combined = causal.unsqueeze(0) & pad.unsqueeze(1)  # [B, S, S]
        # Expand to [B, 1, S, S] and convert to additive mask.
        mask4d = torch.zeros(bsz, 1, seq_len, seq_len, dtype=dtype,
                             device=device)
        mask4d.masked_fill_(~combined.unsqueeze(1), torch.finfo(dtype).min)
        return mask4d

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> CausalLMOutputWithPast:
        K = K if K is not None else self.cfg.K
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 1) Embedding
        h = self.llm.model.embed_tokens(input_ids)

        # 2) Position ids and rotary
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 3) Causal attention mask
        attn_mask_4d = self._build_attention_mask_4d(
            attention_mask, dtype=h.dtype, device=device)

        # 4) Static (non-loop) layers — run once
        static_layers = self.llm.model.layers[:self._n_static]
        for layer in static_layers:
            outputs = layer(h, attention_mask=attn_mask_4d,
                            position_ids=position_ids,
                            past_key_value=None, output_attentions=False,
                            use_cache=False)
            h = outputs[0]

        h_init = h.clone() if self.cfg.input_injection else None

        # 5) Loop layers — run K times
        loop_layers = self.llm.model.layers[self._n_static:]
        for k in range(K):
            for layer in loop_layers:
                outputs = layer(h, attention_mask=attn_mask_4d,
                                position_ids=position_ids,
                                past_key_value=None, output_attentions=False,
                                use_cache=False)
                h = outputs[0]
            if self.cfg.input_injection and h_init is not None:
                h = h + self.cfg.injection_scale * h_init
            if self.cfg.loop_layernorm:
                h = self.loop_ln(h)

        # 6) Final norm + lm_head
        h = self.llm.model.norm(h)
        logits = self.llm.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ----- generate (simple greedy, no KV cache for sanity check) -----

    @torch.no_grad()
    def generate_greedy(self, input_ids, max_new_tokens: int = 128,
                        K: Optional[int] = None,
                        eos_token_id: Optional[int] = None) -> torch.Tensor:
        K = K if K is not None else self.cfg.K
        out = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(out, K=K).logits
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break
        return out

    # ----- utilities -----

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
