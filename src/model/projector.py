"""Vision -> LLM projector. Kept deliberately boring to match LLaVA-1.5."""

import re
import torch
import torch.nn as nn


def build_projector(projector_type: str, vision_hidden: int, llm_hidden: int) -> nn.Module:
    if projector_type == "linear":
        return nn.Sequential(
            nn.Linear(vision_hidden, llm_hidden),
            nn.LayerNorm(llm_hidden),
        )

    m = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if m:
        depth = int(m.group(1))
        layers = [nn.Linear(vision_hidden, llm_hidden)]
        for _ in range(depth - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_hidden, llm_hidden))
        # Terminal LayerNorm keeps the projector output on the same scale as
        # the LLM's native token embeddings. Without it, with a fully frozen
        # LLM the projector tends to collapse to a degenerate solution
        # (v1: constant spaces regardless of input). Adding LN is the fix
        # LLaVA-1.5's projector uses for the same reason.
        layers.append(nn.LayerNorm(llm_hidden))
        return nn.Sequential(*layers)

    raise ValueError(f"unknown projector_type: {projector_type}")
