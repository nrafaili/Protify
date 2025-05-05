import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, AutoModelForSequenceClassification


class gLM2ForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.glm2 = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
    ) -> torch.Tensor:
        assert not output_attentions, (
            "output_attentions=True is not supported by gLM2ForEmbedding."
        )

        out = self.glm2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        return out.last_hidden_state


presets = {
    'GLM2-150': 'tattabio/gLM2_150M',
    'GLM2-650': 'tattabio/gLM2_650M',
    'GLM2-GAIA': 'tattabio/gLM2_650M_embed'
}


def build_glm2_model(preset: str) -> Tuple[gLM2ForEmbedding, AutoTokenizer]:
    model_path = presets[preset]
    model = gLM2ForEmbedding(model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer



def get_glm2_for_training(preset: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == '__main__':
    # py -m src.protify.base_models.glm
    model, tok = build_glm2_model('gLM2-650')
    print(model)
    print(tok)
