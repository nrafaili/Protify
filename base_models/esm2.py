"""
We use the FastESM2 implementation of ESM2, which is exactly equivalent but uses FlashAttention2.
"""
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel


class FastEsmForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.esm(input_ids, attention_mask=attention_mask).last_hidden_state


presets = {
    'ESM2-8': 'facebook/esm2_t6_8M_UR50D',
    'ESM2-35': 'facebook/esm2_t12_35M_UR50D',
    'ESM2-150': 'facebook/esm2_t30_150M_UR50D',
    'ESM2-650': 'facebook/esm2_t33_650M_UR50D',
    'ESM2-3B': 'facebook/esm2_t36_3B_UR50D',
}


def build_esm2_model(preset: str):
    model = FastEsmForEmbedding.from_pretrained(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esm2_model('ESM2-8')
    print(model)
    print(tokenizer)
