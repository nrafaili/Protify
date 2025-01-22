"""
We use the ESM++ implementation of ESMC, which is exactly equivalent but offers batching.
"""
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing

### Tokenization
SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

class EsmSequenceTokenizer(PreTrainedTokenizerFast):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        unk_token="<unk>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        eos_token="<eos>",
        chain_break_token="|",
        **kwargs,
    ):
        all_tokens = SEQUENCE_VOCAB
        token_to_id = {tok: ind for ind, tok in enumerate(all_tokens)}

        # a character-level tokenizer is the same as BPE with no token merges
        bpe = BPE(token_to_id, merges=[], unk_token=unk_token)
        tokenizer = Tokenizer(bpe)
        special_tokens = [
            cls_token,
            pad_token,
            mask_token,
            eos_token,
            chain_break_token,
        ]
        self.cb_token = chain_break_token
        additional_special_tokens = [chain_break_token]

        tokenizer.add_special_tokens(special_tokens)

        # This is where we configure the automatic addition of special tokens when we call
        # tokenizer(text, add_special_tokens=True). Note that you can also configure how two
        # sequences are merged if you want.
        tokenizer.post_processor = TemplateProcessing(  # type: ignore
            single="<cls> $A <eos>",
            special_tokens=[
                ("<cls>", tokenizer.token_to_id("<cls>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # These are a footgun, we never use the `bos` token anywhere so we're just overriding it here.
    @property
    def bos_token(self):
        return self.cls_token

    @property
    def bos_token_id(self):
        return self.cls_token_id

    @property
    def chain_break_token(self):
        return self.cb_token

    @property
    def chain_break_token_id(self):
        return self.convert_tokens_to_ids(self.chain_break_token)

    @property
    def all_token_ids(self):
        return list(range(self.vocab_size))

    @property
    def special_token_ids(self):
        return self.all_special_ids



class ESMplusplusForEmbedding(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> torch.Tensor:
        if output_attentions:
            out = self.esm(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
            return out.last_hidden_state, out.attentions
        else:
            return self.esm(input_ids, attention_mask=attention_mask).last_hidden_state

presets = {
    'ESMC-300': 'Synthyra/ESMplusplus_small',
    'ESMC-600': 'Synthyra/ESMplusplus_large',
}


def build_esmc_model(preset: str):
    model = ESMplusplusForEmbedding(presets[preset]).eval()
    tokenizer = model.esm.tokenizer
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = build_esmc_model('ESMC-300')
    print(model)
    print(tokenizer)
