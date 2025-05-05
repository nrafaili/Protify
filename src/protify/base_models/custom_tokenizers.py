from typing import List, Dict, Union
import torch
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from .FastPLMs.modeling_esm_plusplus import EsmSequenceTokenizer


class BaseSequenceTokenizer:
    def __init__(self, tokenizer: Union[AutoTokenizer, EsmTokenizer, EsmSequenceTokenizer, T5Tokenizer]):
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None.")
        self.tokenizer = tokenizer

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        # Default tokenizer args if not provided
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'max_length')
        kwargs.setdefault('add_special_tokens', True)

        return self.tokenizer(sequences, **kwargs)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, 'pad_token_id')

    @property
    def eos_token_id(self):
        return getattr(self.tokenizer, 'eos_token_id')

    @property
    def cls_token_id(self):
        return getattr(self.tokenizer, 'cls_token_id')

class ANKHTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        pass
    
class ESMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: EsmTokenizer):
        super().__init__(tokenizer)
        pass


class T5TokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: Union[T5Tokenizer, AutoTokenizer]):
        super().__init__(tokenizer)
        pass


class BERTTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: Union[EsmTokenizer]):
        super().__init__(tokenizer)
        pass

class GLMTokenizerWrapper(BaseSequenceTokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.plus_token = "<+>"
        if self.plus_token not in self.tokenizer.vocab:
            print(f"Warning: Token '{self.plus_token}' not found in GLM tokenizer vocabulary.")

    def __call__(self, sequences: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        kwargs.setdefault('return_tensors', 'pt')
        kwargs.setdefault('padding', 'longest')
        kwargs.setdefault('add_special_tokens', True)
        modified_sequences = [self.plus_token + seq for seq in sequences]
        tokenized = self.tokenizer(modified_sequences, **kwargs)

        return tokenized

