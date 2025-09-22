import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

from .base_tokenizer import BaseSequenceTokenizer


presets = {
    'AMPLIFY-120': 'chandar-lab/AMPLIFY_120M',
    'AMPLIFY-350': 'chandar-lab/AMPLIFY_350M', 
    # Base-length variants (context_length = 512)
    'AMPLIFY-120-base': 'chandar-lab/AMPLIFY_120M_base',
    'AMPLIFY-350-base': 'chandar-lab/AMPLIFY_350M_base',
}


class AmplifyTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        tokenized = self.tokenizer(sequences, **kwargs)
        # Convert attention_mask to how AMPLIFY expects it
        tokenized['attention_mask'] = torch.where(
            tokenized['attention_mask'] == 1,
            torch.tensor(0.0),
            torch.tensor(float('-inf'))
        )
        return tokenized


class AmplifyForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.plm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> torch.Tensor:

        out = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        if output_attentions:
            return out.hidden_states[-1].float(), out.attentions
        else:
            return out.hidden_states[-1].float()


def get_amplify_tokenizer(preset: str):
    return AmplifyTokenizerWrapper(AutoTokenizer.from_pretrained(presets[preset], trust_remote_code=True))


def build_amplify_model(preset: str) -> Tuple[AmplifyForEmbedding, AutoTokenizer]:
    model_path = presets[preset]
    model = AmplifyForEmbedding(model_path).eval()
    tokenizer = get_amplify_tokenizer(preset)
    return model, tokenizer


def get_amplify_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    model_path = presets[preset]
    if hybrid:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    else:
        if tokenwise:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, trust_remote_code=True
            ).eval()
    tokenizer = get_amplify_tokenizer(preset)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.amplify
    model, tokenizer = build_amplify_model('AMPLIFY-120')
    print(model)
    print(tokenizer)
    print(tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL'))