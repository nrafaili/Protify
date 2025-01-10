import torch
import torch.nn as nn
from typing import Optional
from transformers import EsmTokenizer
from .esm2 import FastEsmConfig, FastEsmForEmbedding


presets = {
    'random': 'random',
    'random_weights': 'facebook/esm2_t12_35M_UR50D', # default is 35M version
    'random_esm2_8': 'facebook/esm2_t6_8M_UR50D',
    'random_esm2_35': 'facebook/esm2_t12_35M_UR50D',
    'random_esm2_150': 'facebook/esm2_t30_150M_UR50D',
    'random_esm2_650': 'facebook/esm2_t36_650M_UR50D',
    'random_esm2_3B': 'facebook/esm2_t36_3B_UR50D'
}


class RandomModel(nn.Module):
    def __init__(self, config: FastEsmConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.randn(input_ids.shape[0], self.hidden_size)


def build_random_model(preset: str):
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    if preset == 'random':
        model = RandomModel(FastEsmConfig.from_pretrained('facebook/esm2_t12_35M_UR50D'))
    else:
        config = FastEsmConfig.from_pretrained(presets[preset])
        model = FastEsmForEmbedding(config).eval()
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_random_model('random_weights')
    print(model)
    print(tokenizer)