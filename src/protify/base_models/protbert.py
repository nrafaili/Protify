import torch
import torch.nn as nn
from typing import Optional
from transformers import BertModel, EsmTokenizer, BertForSequenceClassification, BertForTokenClassification


class ProtBertForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = BertModel.from_pretrained(model_path, attn_implementation="sdpa")

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
    'ProtBert': 'Rostlab/prot_bert',
    'ProtBert-BFD': 'Rostlab/prot_bert_bfd',
}


def build_protbert_model(preset: str):
    model_path = presets[preset]
    model = ProtBertForEmbedding(model_path).eval()
    tokenizer = EsmTokenizer.from_pretrained('lhallee/no_space_protbert_tokenizer')
    return model, tokenizer


def get_protbert_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = BertModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = model.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.protbert
    model, tokenizer = build_protbert_model('protbert-bfd')
    print(model)
    print(tokenizer)
