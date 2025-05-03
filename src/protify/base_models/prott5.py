import torch
import torch.nn as nn
from typing import Optional
from transformers import T5EncoderModel, T5Tokenizer

from .t5 import T5ForSequenceClassification, T5ForTokenClassification


class Prott5ForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = T5EncoderModel.from_pretrained(model_path)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.plm(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.plm(input_ids, attention_mask=attention_mask).last_hidden_state


presets = {
    'ProtT5': 'Rostlab/prot_t5_xl_half_uniref50-enc',
    'ProtT5-XL-UniRef50-full-prec': 'Rostlab/prot_t5_xl_uniref50',
    'ProtT5-XXL-UniRef50': 'Rostlab/prot_t5_xxl_uniref50',
    'ProtT5-XL-BFD': 'Rostlab/prot_t5_xl_bfd',
    'ProtT5-XXL-BFD': 'Rostlab/prot_t5_xxl_bfd',
}


def build_prott5_model(preset: str):
    model_path = presets[preset]
    model = Prott5ForEmbedding(model_path).eval()
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_prott5_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = T5EncoderModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.prott5
    model, tokenizer = build_prott5_model('ProtT5')
    print(model)
    print(tokenizer)
