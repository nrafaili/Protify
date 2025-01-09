import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


Linear = partial(nn.Linear, bias=False)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, dropout: float = 0.1):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2
        ),
        SwiGLU(),
        nn.Dropout(dropout),
        Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model),
    )