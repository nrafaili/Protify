import torch
from torch import nn
from typing import Optional
from .attention import MultiHeadAttention
from .mlp import swiglu_ln_ffn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, rotary)
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r1 = self.attn(x, attention_mask)
        x = x + r1
        r2 = self.ffn(x)
        x = x + r2
        return x
    

class Transformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_layers: int,
            expansion_ratio: float = 8 / 3,
            dropout: float = 0.1,
            rotary: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, expansion_ratio, dropout, rotary) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
