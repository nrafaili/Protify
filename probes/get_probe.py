from dataclasses import dataclass, field
from typing import List
from .linear_probe import LinearProbe, LinearProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig
#from .crossconv import CrossConv, CrossConvConfig


@dataclass
class ProbeArguments:
    probe_type: str # linear, transformer, crossconv
    tokenwise: bool = field(default=False)
    ### Linear Probe
    input_dim: int = field(default=960)
    hidden_dim: int = field(default=8192)
    dropout: float = field(default=0.2)
    num_labels: int = field(default=2)
    n_layers: int = field(default=1)
    task_type: str = field(default='binary')
    pre_ln: bool = field(default=True)
    ### Transformer Probe
    ff_dim: int = field(default=4096)
    transformer_dropout: float = field(default=0.1)
    classifier_dropout: float = field(default=0.2)
    n_heads: int = field(default=4)
    rotary: bool = field(default=True)
    pooling_types: List[str] = field(default_factory=lambda: ['mean', 'cls'])


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
