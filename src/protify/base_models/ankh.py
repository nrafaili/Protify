import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
from transformers import T5EncoderModel, AutoTokenizer

from .base_tokenizer import BaseSequenceTokenizer
from .t5 import T5ForSequenceClassification, T5ForTokenClassification


presets = {
    'ANKH-Base': 'Synthyra/ANKH_base',
    'ANKH-Large': 'Synthyra/ANKH_large',
    'ANKH2-Large': 'Synthyra/ANKH2_large',
}


class ANKHTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        return tokenized


class AnkhForEmbedding(nn.Module):
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


def get_ankh_tokenizer(preset: str):
    return ANKHTokenizerWrapper(AutoTokenizer.from_pretrained('Synthyra/ANKH_base'))


def build_ankh_model(preset: str):
    model_path = presets[preset]
    model = AnkhForEmbedding(model_path).eval()
    tokenizer = get_ankh_tokenizer(preset)
    return model, tokenizer


def get_ankh_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = T5EncoderModel.from_pretrained(model_path).eval()
    else:
        if tokenwise:
            model = T5ForTokenClassification.from_pretrained(model_path, num_labels=num_labels).eval()
        else:
            model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).eval()
    tokenizer = get_ankh_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.ankh
    model, tokenizer = build_ankh_model('ANKH-Base')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICBBOLLICIIVMLL'))
