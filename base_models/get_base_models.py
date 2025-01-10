from dataclasses import dataclass, field


currently_supported_models = [
    'esm2_8',
    'esm2_35',
    'esm2_150',
    'esm2_650',
    'esm2_3B',
    'random',
    'random_esm2_8',
    'random_esm2_35', # same as random_weights
    'random_esm2_150',
    'random_esm2_650',
    'random_esm2_3B',
    'esmc_300',
    'esmc_600'
]


standard_benchmark = [
    'esm2_8',
    'esm2_35',
    'esm2_150',
    'esm2_650',
    'esm2_3B',
    'esmc_300',
    'esmc_600',
    'random',
    'random_weights'
]


@dataclass
class BaseModelArguments:
    model_names: list[str] = field(default_factory=lambda: standard_benchmark)


def get_base_model(model_name: str):
    if 'random' in model_name:
        from .random import build_random_model
        return build_random_model(model_name)
    elif 'esm2' in model_name:
        from .esm2 import build_esm2_model
        return build_esm2_model(model_name)
    elif 'esmc' in model_name:
        from .esmc import build_esmc_model
        return build_esmc_model(model_name)
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
