import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from typing import List, Optional
from embedder import Pooler


class HybridProbeConfig(PretrainedConfig):
    model_type = "hybrid_probe"
    def __init__(
            self,
            tokenwise: bool = False,
            pooling_types: List[str] = ['mean', 'cls'],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenwise = tokenwise
        self.pooling_types = pooling_types


class HybridProbe(PreTrainedModel):
    config_class = HybridProbeConfig
    def __init__(self, config: HybridProbeConfig, model: nn.Module, probe: nn.Module):
        super().__init__(config)
        self.config = config
        self.tokenwise = config.tokenwise
        self.pooler = Pooler(config.pooling_types)
        self.model = model
        self.probe = probe

    def forward(
            self,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if not self.tokenwise:
            x = self.pooler(x, attention_mask)
        return self.probe(x, labels=labels)
