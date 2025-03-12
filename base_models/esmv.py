import torch
import torch.nn as nn
from typing import Optional
from torchmetrics.functional import pairwise_cosine_similarity
from .FastPLMs.modeling_esm_plusplus import ESMplusplusModel, ESMplusplusConfig, ESMplusplusOutput


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


class ESMV(ESMplusplusModel):
    config_class = ESMplusplusConfig
    def __init__(self, config: ESMplusplusConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.cls_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.eos_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.sequence_head = LMHead(config.hidden_size, self.vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()

    def get_cls_vecs(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.cls_proj(output.last_hidden_state[:, 0, :])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ESMplusplusOutput:
        """Forward pass for sequence classification.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for classification
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            ESMplusplusOutput containing loss, logits, and hidden states
        """
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        x = output.last_hidden_state
        logits = self.sequence_head(x)

        cls_features = x[:, 0, :]
        eos_positions = attention_mask.sum(dim=1).int() - 1  # (batch_size,)
        batch_indices = torch.arange(x.size(0), device=x.device)
        eos_features = x[batch_indices, eos_positions, :]
        cls_vecs = self.cls_proj(cls_features)
        eos_vecs = self.eos_proj(eos_features)
        sims = pairwise_cosine_similarity(cls_vecs, eos_vecs)
        targets = torch.arange(sims.size(0), device=sims.device)

        if labels is not None:
            loss = self.ce_loss(logits.view(-1, self.vocab_size), labels.view(-1))
            loss += self.ce_loss(sims, targets)

        return ESMplusplusOutput(
            loss=loss,
            logits=(logits, cls_vecs, eos_vecs),
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class ESMVForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = ESMV.from_pretrained(model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        return self.esm.get_cls_vecs(input_ids, attention_mask=attention_mask)


presets = {
    'ESMV': 'Synthyra/ESMV',
}


def build_esmv_model(preset: str):
    model = ESMVForEmbedding(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esmv_model('ESMV')
    print(model)
    print(tokenizer)
