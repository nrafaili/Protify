import os
import argparse
import yaml
import torch
import numpy as np
from probes.get_probe import ProbeArguments
from base_models.get_base_models import BaseModelArguments, get_base_model, get_tokenizer
from data.hf_data import HFDataArguments, get_hf_data
from embedder import EmbeddingArguments, Embedder
from probes.trainers import TrainerArguments, train_probe


class MainProcess:
    def __init__(self, full_args):
        self.data_args = HFDataArguments(**full_args)
        self.embedding_args = EmbeddingArguments(**full_args)
        self.model_args = BaseModelArguments(**full_args)
        self.probe_args = ProbeArguments(**full_args)
        self.trainer_args = TrainerArguments(**full_args)
        self.get_datasets()

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

    def run_probes(self):
        model_names = self.model_args.model_names
        probe_args = self.probe_args
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
                    test_embedding = c.fetchone()[0]
                    test_embedding = torch.tensor(np.frombuffer(test_embedding, dtype=np.float32).reshape(1, -1))
                emb_dict = None

            else:
                from safetensors.torch import safe_open
                save_path = os.path.join(self.embedding_args.save_dir, f'{model_name}_{full}.safetensors')
                emb_dict = {}
                with safe_open(save_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        emb_dict[key] = f.get_tensor(key).clone()
                    test_embedding = emb_dict[self.all_seqs[0]].reshape(1, -1)

            if full:
                test_embedding = test_embedding.reshape(seq_len, -1)
            input_dim = test_embedding.shape[-1]
            probe_args.input_dim = input_dim

            for dataset in self.datasets:
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                probe_args.num_labels = num_labels
                probe_args.label_type = label_type
                tokenizer = get_tokenizer(model_name)
                train_probe(
                    self.trainer_args,
                    self.embedding_args,
                    probe_args,
                    tokenizer,
                    train_set,
                    valid_set,
                    test_set,
                    model_name,
                    emb_dict=emb_dict,
                    ppi=ppi,
                )

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script with arguments mirroring the provided YAML settings.")
    # get settings from yaml
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to the YAML file.")

    # ----------------- DataArguments ----------------- #
    parser.add_argument("--data_dirs", nargs="+", default=["DeepLoc-2"], help="List of data directories.")
    parser.add_argument("--delimiter", default=",", help="Delimiter for data.")
    parser.add_argument("--col_names", nargs="+", default=["seqs", "labels"], help="Column names.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--trim", action="store_true", default=False, help="Whether to trim sequences (default: False).")

    # ----------------- BaseModelArguments ----------------- #
    parser.add_argument("--model_names", nargs="+",
                        default=["ESM2-8", "ESMC-300", "Random", "Random-Transformer"], help="List of model names to use.")

    # ----------------- ProbeArguments ----------------- #
    parser.add_argument("--probe_type", choices=["linear", "transformer", "crossconv"], default="linear", help="Type of probe.")
    parser.add_argument("--tokenwise", action="store_true", default=False, help="Tokenwise probe (default: False).")
    parser.add_argument("--hidden_dim", type=int, default=8192, help="Hidden dimension size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--pre_ln", action="store_false", default=True,
                        help="Disable pre-layernorm (default: enabled). Use --pre_ln to toggle off.")
    parser.add_argument("--ff_dim", type=int, default=4096, help="Feed-forward dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for the transformer layers.")
    parser.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout rate for the classifier.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads in multi-head attention.")
    parser.add_argument("--rotary", action="store_false", default=True,
                        help="Disable rotary embeddings (default: enabled). Use --rotary to toggle off.")
    parser.add_argument("--pooling_types", nargs="+", default=["cls"], help="Pooling types to use.")

    # ----------------- EmbeddingArguments ----------------- #
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", default=False, help="Whether to download embeddings (default: False).")
    parser.add_argument("--download_dir", default="Synthyra/mean_pooled_embeddings", help="Directory to download embeddings to.")
    parser.add_argument("--matrix_embed", action="store_true", default=False, help="Use matrix embedding (default: False).")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean"], help="Pooling types for embeddings.")
    parser.add_argument("--save_embeddings", action="store_true", default=False, help="Save computed embeddings (default: False).")
    parser.add_argument("--embed_dtype", default="float16", help="Data type for embeddings.")
    parser.add_argument("--sql", action="store_true", default=False, help="Whether to use SQL storage (default: False).")
    parser.add_argument("--save_dir", default="embeddings", help="Directory to save embeddings.")

    # ----------------- TrainerArguments ----------------- #
    parser.add_argument("--output_dir", default="Synthyra/probes", help="Directory to save probes.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")

    if args.yaml_path is not None:
        with open(args.yaml_path, 'r') as file: 
            settings = yaml.safe_load(file)
        return settings
    else:
        return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
