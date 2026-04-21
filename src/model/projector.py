"""Vision -> LLM projector. Kept deliberately boring to match LLaVA-1.5."""

import re
import torch
import torch.nn as nn


def build_projector(projector_type: str, vision_hidden: int, llm_hidden: int) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(vision_hidden, llm_hidden)

    m = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if m:
        depth = int(m.group(1))
        layers = [nn.Linear(vision_hidden, llm_hidden)]
        for _ in range(depth - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_hidden, llm_hidden))
        return nn.Sequential(*layers)

    raise ValueError(f"unknown projector_type: {projector_type}")
