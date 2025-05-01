"""
We use the ESM++ implementation of ESMC, which is exactly equivalent but offers batching.
"""
import torch
import torch.nn as nn
from typing import Optional
from .FastPLMs.modeling_esm_plusplus import ESMplusplusModel, ESMplusplusForSequenceClassification, ESMplusplusForTokenClassification


class ESMplusplusForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = ESMplusplusModel.from_pretrained(model_path)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.esm(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.esm(input_ids, attention_mask=attention_mask).last_hidden_state

presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
}


def build_esmc_model(preset: str):
    model = ESMplusplusForEmbedding(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


def get_esmc_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = ESMplusplusModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = ESMplusplusForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = ESMplusplusForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
