import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from transformers.modeling_outputs import ModelOutput
from model_components.transformer import TransformerBlock, TransformerForMaskedLM, TransformerConfig
from model_components.attention import AttentionPooler


class CAMPConfig(PretrainedConfig):
    model_type = "CAMP"
    def __init__(
            self,
            hidden_size: int = 768,
            n_heads: int = 12,
            n_layers: int = 4,
            expansion_ratio: float = 2.0,
            dropout: float = 0.1,
            d_pooled: int = 1,
            seq_vocab_size: int = 64,
            ann_vocab_size: int = 32000,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.d_pooled = d_pooled
        self.seq_vocab_size = seq_vocab_size
        self.ann_vocab_size = ann_vocab_size


@dataclass
class CAMPOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None


class CAMP(PreTrainedModel):
    config_class = CAMPConfig
    def __init__(self, config: CAMPConfig):
        super().__init__(config)
        self.config = config
        self.esm = AutoModel.from_pretrained('Synthyra/ESMplusplus_small', trust_remote_code=True, torch_dtype=torch.float16)
        for _, param in self.esm.named_parameters():
            param.requires_grad = False

        self.esm_proj = nn.Linear(self.esm.config.hidden_size, config.hidden_size)
        self.t1 = TransformerBlock(config.hidden_size, config.n_heads, config.expansion_ratio, config.dropout, True)
        self.t2 = TransformerBlock(config.hidden_size, config.n_heads, config.expansion_ratio, config.dropout, True)
        self.sequence_lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.seq_vocab_size),
        )
        self.esm_pooler = AttentionPooler(config.hidden_size, config.d_pooled, config.n_heads)

        at_config = TransformerConfig(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            vocab_size=config.ann_vocab_size+1,
            expansion_ratio=config.expansion_ratio,
            dropout=config.dropout,
            rotary=True,
        )
        self.at = TransformerForMaskedLM(at_config)
        self.at_pooler = AttentionPooler(config.hidden_size, config.d_pooled, config.n_heads)
        self.ce_loss = nn.CrossEntropyLoss()

    def seq_vector_inference(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # (b, L) -> (b, L, d) -> (b, d)
        esm_state = self.esm(input_ids, attention_mask).last_hidden_state.float()
        esm_state = self.esm_proj(esm_state)
        if attention_mask is not None:
            batch_size, seq_len = attention_mask.shape
            attention_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
        esm_state = self.t1(esm_state, attention_mask)
        esm_state = self.t2(esm_state, attention_mask)
        return self.esm_pooler(esm_state).squeeze(1)

    def forward(
            self,
            seq_input_ids: torch.Tensor,
            ann_input_ids: torch.Tensor,
            seq_attention_mask: Optional[torch.Tensor] = None,
            seq_labels: Optional[torch.Tensor] = None,
            ann_attention_mask: Optional[torch.Tensor] = None,
            ann_labels: Optional[torch.Tensor] = None,
    ) -> CAMPOutput:
        loss = 0
        esm_state = self.esm(seq_input_ids, seq_attention_mask).last_hidden_state.float()
        esm_state = self.esm_proj(esm_state)
        if seq_attention_mask is not None:
            batch_size, seq_len = seq_attention_mask.shape
            seq_attention_mask = seq_attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len).bool()
        esm_state = self.t1(esm_state, seq_attention_mask)
        seq_logits = self.sequence_lm_head(esm_state)
        if seq_labels is not None:
            loss += self.ce_loss(seq_logits.view(-1, self.config.seq_vocab_size), seq_labels.view(-1))

        esm_state = self.t2(esm_state, seq_attention_mask)
        seq_pooled = self.esm_pooler(esm_state).squeeze(1)

        ann_output = self.at(input_ids=ann_input_ids, attention_mask=ann_attention_mask, labels=ann_labels, return_preds=True)
        ann_preds = ann_output.logits
        if ann_labels is not None:
            loss += ann_output.loss

        ann_state = ann_output.last_hidden_state
        ann_pooled = self.at_pooler(ann_state).squeeze(1)

        sims = pairwise_cosine_similarity(seq_pooled, ann_pooled)  # (B, B)
        targets = torch.arange(len(seq_pooled), device=seq_pooled.device) # (B,)
        loss += self.ce_loss(sims, targets)

        return CAMPOutput(
            loss=loss,
            logits=(seq_logits.argmax(dim=-1), ann_preds, seq_pooled, ann_pooled, seq_labels, ann_labels),
        )


class CAMPForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.camp = CAMP.from_pretrained(model_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.camp.seq_vector_inference(input_ids, attention_mask=attention_mask)


presets = {
    'camp_a': 'lhallee/camp_1_25',
    'camp_b': 'lhallee/camp_1_25_2_epoch',
    'camp_c': 'lhallee/camp_1_25_3_epoch',
    'camp_d': 'lhallee/camp_1_25_3-5_epoch',
    'camp_e': 'lhallee/camp_1_25_5_epoch',
}


def build_camp_model(preset: str):
    model = CAMPForEmbedding(presets[preset]).eval()
    tokenizer = model.camp.esm.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model = CAMPForEmbedding('lhallee/camp_1_25_5_epoch')
    print(model)
