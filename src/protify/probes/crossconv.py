# TODO
import torch
from torch import nn
from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class CrossConvConfig(PretrainedConfig):
    model_type = "crossconv"
    def __init__(
            self,
            input_dim: int = 768,
            hidden_dim: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.task_type = task_type
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.pre_ln = pre_ln


class CrossConv(PreTrainedModel):
    pass
