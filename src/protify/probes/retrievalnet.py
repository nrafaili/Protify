import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from model_components.attention import AttentionPooler, Linear
from model_components.transformer import PTransformer
from .losses import get_loss_fct


class RetrievalNetConfig(PretrainedConfig):
    model_type = "retrievalnet"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 512,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            n_heads: int = 4,
            task_type: str = 'singlelabel',
            expansion_ratio: float = 8 / 3,
            **kwargs,
    ):
        super().__init__(**kwargs)
        assert task_type != 'regression' or num_labels == 1, "Regression task must have exactly one label"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.n_heads = n_heads


class RetrievalNetForSequenceClassification(PreTrainedModel):
    config_class = RetrievalNetConfig
    def __init__(self, config: RetrievalNetConfig):
        super().__init__(config)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.transformer = PTransformer(
            hidden_size=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=config.rotary,
        )
        self.label_map = AttentionPooler(
            hidden_size=config.hidden_dim,
            n_tokens=config.num_labels,
            n_heads=config.n_heads,
        )
        self.output_proj = nn.Linear(config.hidden_dim, 1)
        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> SequenceClassifierOutput:
        x = self.input_proj(embeddings) # (bs, seq_len, hidden_dim)
        x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_dim)
        x, attentions = self.label_map(x, attention_mask, output_attentions) # (bs, num_labels, hidden_dim)
        logits = self.output_proj(x).squeeze(-1) # (bs, num_labels)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=attentions
        )


class RetrievalNetForTokenClassification(PreTrainedModel):
    config_class = RetrievalNetConfig
    def __init__(self, config: RetrievalNetConfig):
        super().__init__(config)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.transformer = PTransformer(
            hidden_size=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=config.rotary,
        )
        self.Wq = Linear(config.hidden_dim, config.hidden_dim)
        self.Pk = nn.Parameter(torch.randn(1, config.hidden_dim, config.num_labels))

        self.num_labels = config.num_labels
        self.task_type = config.task_type
        self.loss_fct = get_loss_fct(config.task_type)

    def forward(
            self,
            embeddings,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> SequenceClassifierOutput:
        x = self.input_proj(embeddings) # (bs, seq_len, hidden_dim)
        x = self.transformer(x, attention_mask) # (bs, seq_len, hidden_dim)
        
        q = self.Wq(x) # (bs, seq_len, hidden_dim)
        k = self.Pk # (bs, hidden_dim, num_labels)

        attn_bias = torch.ones(x.size(1), self.num_labels, device=x.device)
        if attention_mask is not None:
            attn_bias = attn_bias.masked_fill(attention_mask == 0, float("-inf"))
        else:
            attn_bias = attention_mask + attn_bias

        logits = torch.matmul(q, k) # (bs, seq_len, num_labels)

        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.flatten(), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )