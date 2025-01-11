from typing import List, Tuple
from datasets import Dataset
from dataclasses import field

from base_models.get_base_models import BaseModelArguments, get_base_model
from data.hf_data import HFDataArguments, get_hf_data
from embedder import EmbeddingArguments, Embedder


class MainProcess:
    all_seqs: List[str]
    datasets: List[Tuple[Dataset, Dataset, Dataset, int, str]]
    data_args: HFDataArguments
    embedding_args: EmbeddingArguments
    model_args: BaseModelArguments

    def __init__(self):
        pass

    def get_datasets(self):
        self.datasets, self.all_seqs = get_hf_data(self.data_args)

    def save_embeddings_to_disk(self):
        self.embedding_args.save_embeddings = True
        embedder = Embedder(self.embedding_args)
        model_names = self.model_args.model_names
        for model_name in model_names:
            print(f'Embedding sequences with {model_name}')
            model, tokenizer = get_base_model(model_name)
            _ = embedder(model_name, model, tokenizer) 

    def find_best_scikit_model(self):
        pass

    def cross_validate_scikit_model(self):
        pass

    def scikit_workflow(self):
        production = False # TODO: make this a setting
        pass

    def load_probe(self):
        pass

    def run_probe(self):
        pass

    def init_hybrid_probe(self):
        # freeze base model
        # train probe
        self.run_probe()
        # unfreeze base model
        # train base model
        pass

    def init_lora_model(self):
        pass

    def run_lora_model(self):
        pass

    def get_full_finetuning_model(self):
        pass

    def run_full_finetuning_model(self):
        pass

    
