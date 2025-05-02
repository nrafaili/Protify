import torch
from torch import nn
from typing import Optional
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from .attention import MultiHeadAttention
from .mlp import swiglu_ln_ffn


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = False,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, n_heads, rotary)
        self.ffn = swiglu_ln_ffn(hidden_size, expansion_ratio, dropout)

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
            hidden_size: int,
            n_heads: int,
            n_layers: int,
            expansion_ratio: float = 8 / 3,
            dropout: float = 0.1,
            rotary: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, expansion_ratio, dropout, rotary) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class TransformerConfig(PretrainedConfig):
    model_type = "transformer"
    def __init__(
        self,
        hidden_size: int = 512,
        n_heads: int =  8,
        n_layers: int = 12,
        vocab_size: int = 32000,
        expansion_ratio: float = 8 / 3,
        dropout: float = 0.1,
        rotary: bool = True,
    ):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.rotary = rotary
        self.vocab_size = vocab_size


@dataclass
class TransformerOutput(ModelOutput):
    """Output type for ESM++ models."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class TransformerForMaskedLM(PreTrainedModel):
    config_class = TransformerConfig
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = Transformer(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=config.rotary,
        )
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size),
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_preds: bool = True,
    ) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.transformer(x, attention_mask)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
        return TransformerOutput(
            loss=loss,
            logits=logits.argmax(dim=-1) if return_preds else logits,
            last_hidden_state=x,
        )
