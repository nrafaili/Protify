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
    'ProtBert',
    'ProtBert-BFD',
    'ProtT5',
    'ProtT5-XL-UniRef50-full-prec',
    'ProtT5-XXL-UniRef50',
    'ProtT5-XL-BFD',
    'ProtT5-XXL-BFD',
    'ANKH-Base',
    'ANKH-Large',
    'ANKH2-Large',
    'GLM2-150',
    'GLM2-650',
    'GLM2-GAIA',
]

standard_models = [
    'ESM2-8',
    'ESM2-35',
    'ESM2-150',
    'ESM2-650',
    'ESM2-3B',
    'ESMC-300',
    'ESMC-600',
    'ProtBert',
    'ProtT5',
    'GLM2-150',
    'GLM2-650',
    'GLM2-GAIA',
    'ANKH-Base',
    'ANKH-Large',
    'ANKH2-Large',
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
    elif 'protbert' in model_name.lower():
        from .protbert import build_protbert_model
        return build_protbert_model(model_name)
    elif 'prott5' in model_name.lower():
        from .prott5 import build_prott5_model
        return build_prott5_model(model_name)
    elif 'ankh' in model_name.lower():
        from .ankh import build_ankh_model
        return build_ankh_model(model_name)
    elif 'glm' in model_name.lower():
        from .glm import build_glm2_model
        return build_glm2_model(model_name)
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
        from custom_tokenizers import ESMTokenizerWrapper
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        return ESMTokenizerWrapper(tokenizer)
    elif 'esmc' in model_name.lower() or 'camp' in model_name.lower() or 'esmv' in model_name.lower():
        from .FastPLMs.modeling_esm_plusplus import EsmSequenceTokenizer
        return EsmSequenceTokenizer()
    elif 'protbert' in model_name.lower():
        from transformers import EsmTokenizer
        from custom_tokenizers import BERTTokenizerWrapper
        tokenizer = EsmTokenizer.from_pretrained('lhallee/no_space_protbert_tokenizer')
        return BERTTokenizerWrapper(tokenizer)
    elif 'prott5' in model_name.lower():
        from transformers import T5Tokenizer
        from custom_tokenizers import T5TokenizerWrapper
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
        return T5TokenizerWrapper(tokenizer)
    elif 'ankh' in model_name.lower():
        from transformers import AutoTokenizer
        from custom_tokenizers import ANKHTokenizerWrapper
        tokenizer = AutoTokenizer.from_pretrained('Synthyra/ANKH-Base')
        return ANKHTokenizerWrapper(tokenizer)
    elif 'glm' in model_name.lower():
        from transformers import AutoTokenizer
        from custom_tokenizers import GLMTokenizerWrapper
        tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_150M')
        return GLMTokenizerWrapper(tokenizer)
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
