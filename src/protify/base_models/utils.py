import torch.nn as nn
from peft import LoraConfig, LoraModel


def wrap_lora(module: nn.Module, r: int, lora_alpha: float, lora_dropout: float) -> nn.Module:
    # these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
    target_modules=["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if 'classifier' in name.lower():
            param.requires_grad = True
    return module
