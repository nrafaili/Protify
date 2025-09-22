# src/protify/base_models/saprot.py

import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

from .base_tokenizer import BaseSequenceTokenizer


presets = {
    "SaProt-35M-AF2": "westlake-repl/SaProt_35M_AF2",
    "SaProt-650M-AF2": "westlake-repl/SaProt_650M_AF2",
}


class SaProtTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", "longest")
        kwargs.setdefault("add_special_tokens", True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class SaProtForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        if output_attentions:
            out = self.plm(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def get_saprot_tokenizer(preset: str):
    return SaProtTokenizerWrapper(AutoTokenizer.from_pretrained(presets[preset]))


def build_saprot_model(preset: str) -> Tuple[SaProtForEmbedding, AutoTokenizer]:
    model_path = presets[preset]
    model = SaProtForEmbedding(model_path).eval()
    tokenizer = get_saprot_tokenizer(preset)
    return model, tokenizer


def get_saprot_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = get_saprot_tokenizer(preset)
    return model, tokenizer


if __name__ == "__main__":
    # py -m src.protify.base_models.saprot
    model, tokenizer = build_saprot_model("SaProt-650M-AF2")
    print(model)
    print(tokenizer)
    print(tokenizer("MdEvVpQpLrVyQdYaKv"))
