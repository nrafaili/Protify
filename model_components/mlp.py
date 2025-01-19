import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


Linear = partial(nn.Linear, bias=False)


def intermediate_correction_fn(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(hidden_size: int, expansion_ratio: float, dropout: float = 0.1):
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        Linear(
            hidden_size, intermediate_correction_fn(expansion_ratio, hidden_size) * 2
        ),
        SwiGLU(),
        nn.Dropout(dropout),
        Linear(intermediate_correction_fn(expansion_ratio, hidden_size), hidden_size),
    )
