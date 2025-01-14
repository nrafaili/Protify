import os
from typing import List, Tuple
from datasets import Dataset
from dataclasses import field
from probes.get_probe import ProbeArguments
from base_models.get_base_models import BaseModelArguments, get_base_model
from data.hf_data import HFDataArguments, get_hf_data
from embedder import EmbeddingArguments, Embedder
from probes.get_probe import get_probe


class MainProcess:
    all_seqs: List[str]
    datasets: List[Tuple[Dataset, Dataset, Dataset, int, str]]
    data_args: HFDataArguments
    embedding_args: EmbeddingArguments
    model_args: BaseModelArguments
    probe_args: ProbeArguments

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
        probe = get_probe(self.probe_args)
        return probe

    def run_probes(self):
        model_names = self.model_args.model_names

        for model_name in model_names:
            sql = self.embedding_args.sql
            max_len = self.data_args.max_len
            seq = self.all_seqs[0]
            full = self.embedding_args.matrix_embed
            if len(seq) < max_len - 2:
                seq_len = max_len
            else:
                seq_len = len(seq) + 2
            if sql:
                import sqlite3
                save_path = os.path.join(self.embedding_args.save_dir, f'{model_name}_{full}.db')
                with sqlite3.connect(save_path) as conn:
                    c = conn.cursor()
                    c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (self.all_seqs[0],))
                    embedding = c.fetchone()[0]

            else:
                from safetensors.torch import safe_open
                save_path = os.path.join(self.embedding_args.save_dir, f'{model_name}_{full}.safetensors')
                with safe_open(save_path, framework="pt", device="cpu") as f:
                    embedding = f.get_tensor(self.all_seqs[0]).clone()

            if full:
                embedding = embedding.reshape(seq_len, -1)
            input_dim = embedding.shape[1]
            
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

    
