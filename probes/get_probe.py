from dataclasses import dataclass, field
from typing import List
from .linear_probe import LinearProbe, LinearProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig
#from .crossconv import CrossConv, CrossConvConfig


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'linear', # valid options: linear, transformer, crossconv
            tokenwise: bool = False,
            ### Linear Probe
            input_dim: int = 960,
            hidden_dim: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'binary',
            pre_ln: bool = True,
            ### Transformer Probe
            ff_dim: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            n_heads: int = 4,
            rotary: bool = True,
            pooling_types: List[str] = field(default_factory=lambda: ['mean', 'cls']),
            **kwargs,
            ### CrossConv
            # TODO
    ):
        self.probe_type = probe_type
        self.tokenwise = tokenwise
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.ff_dim = ff_dim
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.n_heads = n_heads
        self.rotary = rotary
        self.pooling_types = pooling_types


def get_probe(args: ProbeArguments):
    if args.probe_type == 'linear' and not args.tokenwise:
        config = LinearProbeConfig(args)
        return LinearProbe(config)
    elif args.probe_type == 'transformer' and not args.tokenwise:
        config = TransformerProbeConfig(args)
        return TransformerForSequenceClassification(config)
    elif args.probe_type == 'transformer' and args.tokenwise:
        config = TransformerProbeConfig(args)
        return TransformerForTokenClassification(config)
    elif args.probe_type == 'crossconv' and args.tokenwise:
        # TODO
        pass
        # config = CrossConvConfig(args)
        # return CrossConv(config)
    else:
        raise ValueError(f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}")
