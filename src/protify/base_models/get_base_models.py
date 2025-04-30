from dataclasses import dataclass


currently_supported_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'Random',
    'Random-Transformer',
    'Random-ESM2-8',
    'Random-ESM2-35', # same as Random-Transformer
    'Random-ESM2-150',
    'Random-ESM2-650',
    'ESMC-300',
    'ESMC-600',
    'ESM2-diff-150',
    'ESM2-diffAV-150',
]

standard_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'ESMC-300',
    'ESMC-600',
    'Random',
    'Random-Transformer',
]

experimental_models = []


@dataclass
class BaseModelArguments:
    def __init__(self, model_names: list[str] = None, **kwargs):
        if model_names[0] == 'standard':
            self.model_names = standard_models
        elif 'exp' in model_names[0].lower():
            self.model_names = experimental_models
        else:
            self.model_names = model_names


def get_base_model(model_name: str):
    if 'random' in model_name.lower():
        from .random import build_random_model
        return build_random_model(model_name)
    elif 'esm2' in model_name.lower():
        from .esm2 import build_esm2_model
        return build_esm2_model(model_name)
    elif 'esmc' in model_name.lower():
        from .esmc import build_esmc_model
        return build_esmc_model(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_base_model_for_training(model_name: str, tokenwise: bool = False, num_labels: int = None, hybrid: bool = False):
    if 'esm2' in model_name.lower():
        from .esm2 import get_esm2_for_training
        return get_esm2_for_training(model_name, tokenwise, num_labels, hybrid)
    elif 'esmc' in model_name.lower():
        from .esmc import get_esmc_for_training
        return get_esmc_for_training(model_name, tokenwise, num_labels, hybrid)
    else:
        raise ValueError(f"Model {model_name} not supported")


def get_tokenizer(model_name: str):
    if 'esm2' in model_name.lower() or 'random' in model_name.lower():
        from transformers import EsmTokenizer
        return EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    elif 'esmc' in model_name.lower() or 'camp' in model_name.lower() or 'esmv' in model_name.lower():
        from .FastPLMs.modeling_esm_plusplus import EsmSequenceTokenizer
        return EsmSequenceTokenizer()
    else:
        raise ValueError(f"Model {model_name} not supported")


if __name__ == '__main__':
    ### This will download all standard models
    from torchinfo import summary
    from ..utils import clear_screen
    args = BaseModelArguments()
    for model_name in args.model_names:
        model, tokenizer = get_base_model(model_name)
        print(f'Downloaded {model_name}')
        tokenized = tokenizer('MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLICIIVMLL', return_tensors='pt').input_ids
        summary(model, input_data=tokenized)
        clear_screen()
