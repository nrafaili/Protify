"""
We use the ESM++ implementation of ESMC, which is exactly equivalent but offers batching.
"""
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel


class ESMplusplusForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm(input_ids, attention_mask=attention_mask).last_hidden_state

presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
}


def build_esmc_model(preset: str):
    model = ESMplusplusForEmbedding.from_pretrained(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
