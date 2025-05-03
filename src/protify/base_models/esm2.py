"""
We use the FastESM2 implementation of ESM2, which is exactly equivalent but uses FlashAttention2.
"""
import torch
import torch.nn as nn
from typing import Optional
from .FastPLMs.modeling_fastesm import FastEsmModel, FastEsmForSequenceClassification, FastEsmForTokenClassification


class FastEsmForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = FastEsmModel.from_pretrained(model_path)

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
    'ESM2-8': 'Synthyra/ESM2-8M',
    'ESM2-35': 'Synthyra/ESM2-35M',
    'ESM2-150': 'Synthyra/ESM2-150M',
    'ESM2-650': 'Synthyra/ESM2-650M',
    'ESM2-3B': 'Synthyra/ESM2-3B',
    'ESM2-diff-150': 'Synthyra/esm_diff_150',
    'ESM2-diffAV-150': 'Synthyra/esm_diff_av_150_41000'
}


def build_esm2_model(preset: str):
    model = FastEsmForEmbedding(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


def get_esm2_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = FastEsmModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = FastEsmForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = FastEsmForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esm2_model('ESM2-8')
    print(model)
    print(tokenizer)
