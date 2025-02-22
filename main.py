import os
import argparse
import yaml
import torch
import numpy as np
from types import SimpleNamespace
from probes.get_probe import ProbeArguments
from base_models.get_base_models import BaseModelArguments, get_tokenizer
from data.hf_data import HFDataArguments, get_hf_data
from probes.trainers import TrainerArguments, train_probe
from embedder import EmbeddingArguments, Embedder
from logger import MetricsLogger, log_method_calls
from utils import torch_load


class MainProcess(MetricsLogger):
    def __init__(self, full_args, GUI=False):
        super().__init__(full_args)
        self.full_args = full_args
        if not GUI:
            self.start_log_main()

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            #"int8": torch.int8,
        }

    @log_method_calls
    def apply_current_settings(self):
        self.full_args.embed_dtype = self.dtype_map[self.full_args.embed_dtype]
        self.data_args = HFDataArguments(**self.full_args.__dict__)
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
        self.model_args = BaseModelArguments(**self.full_args.__dict__)
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        self.trainer_args = TrainerArguments(**self.full_args.__dict__)
        self.logger_args = SimpleNamespace(**self.full_args.__dict__)

    @log_method_calls
    def get_datasets(self):
        self.datasets, self.all_seqs = get_hf_data(self.data_args)

    @log_method_calls
    def save_embeddings_to_disk(self):
        self.embedding_args.save_embeddings = True
        embedder = Embedder(self.embedding_args, self.all_seqs)
        model_names = self.model_args.model_names
        for model_name in model_names:
            _ = embedder(model_name)

    @log_method_calls
    def run_probes(self): # TODO refactor to run_nn_probe
        model_names = self.model_args.model_names
        probe_args = self.probe_args
        
        # Log the combinations we're going to process
        total_combinations = len(model_names) * len(self.datasets)
        self.logger.info(f"Processing {total_combinations} model/dataset combinations")
        
        for model_name in model_names:
            self.logger.info(f"Processing model: {model_name}")
            sql = self.embedding_args.sql
            max_length = self.data_args.max_length
            test_seq = self.all_seqs[0]
            full = self.embedding_args.matrix_embed

            if len(test_seq) > max_length - 2:
                test_seq_len = max_length
            else:
                test_seq_len = len(test_seq) + 2
    
            if sql:
                import sqlite3
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{full}.db')
                with sqlite3.connect(save_path) as conn:
                    c = conn.cursor()
                    c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (test_seq,))
                    test_embedding = c.fetchone()[0]
                    test_embedding = torch.tensor(np.frombuffer(test_embedding, dtype=np.float32).reshape(1, -1))
                emb_dict = None
            else:
                save_path = os.path.join(self.embedding_args.embedding_save_dir, f'{model_name}_{full}.pth')
                emb_dict = torch_load(save_path)
                test_embedding = emb_dict[test_seq].reshape(1, -1)

            if full:
                test_embedding = test_embedding.reshape(test_seq_len, -1)
            input_dim = test_embedding.shape[-1]
            tokenizer = get_tokenizer(model_name)

            for data_name, dataset in self.datasets.items():
                self.logger.info(f"Processing dataset: {data_name}")
                train_set, valid_set, test_set, num_labels, label_type, ppi = dataset
                print(input_dim)
                if ppi:
                    probe_args.input_dim = input_dim * 2
                else:
                    probe_args.input_dim = input_dim
            
                probe_args.num_labels = num_labels
                probe_args.task_type = label_type
                self.trainer_args.task_type = label_type
                self.logger.info(f'Training probe for {data_name} with {model_name}')
                valid_metrics, test_metrics = train_probe(
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
                self.log_metrics(data_name, model_name, valid_metrics, split_name='valid')
                self.log_metrics(data_name, model_name, test_metrics, split_name='test')

    @log_method_calls
    def init_hybrid_probe(self):
        # freeze base model
        # train probe
        # unfreeze base model
        # train base model
        pass

    @log_method_calls
    def init_lora_model(self):
        pass

    @log_method_calls
    def run_lora_model(self):
        pass

    @log_method_calls
    def get_full_finetuning_model(self):
        pass

    @log_method_calls
    def run_full_finetuning_model(self):
        pass

    @log_method_calls
    def find_best_scikit_model(self):
        pass

    @log_method_calls
    def run_scikit_model(self):
        production = False # TODO: make this a setting
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script with arguments mirroring the provided YAML settings.")
    # ----------------- ID ----------------- #
    parser.add_argument("--hf_username", default="Synthyra", help="Hugging Face username.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token.")
    parser.add_argument("--synthyra_api_key", default=None, help="Synthyra API key.")
    parser.add_argument("--wandb_api_key", default=None, help="Wandb API key.")

    # ----------------- Paths ----------------- #
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to the YAML file.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to the log directory.")
    parser.add_argument("--results_dir", type=str, default="results", help="Path to the results directory.")
    parser.add_argument("--data_paths", nargs="+", default=["DeepLoc-2"], help="List of data directories.") # TODO rename to data_names
    parser.add_argument("--model_save_dir", default="weights", help="Directory to save models.")
    parser.add_argument("--embedding_save_dir", default="embeddings", help="Directory to save embeddings.")
    parser.add_argument("--download_dir", default="Synthyra/mean_pooled_embeddings", help="Directory to download embeddings to.")
    parser.add_argument("--replay_path", type=str, default=None, help="Path to the replay file.")

    # ----------------- DataArguments ----------------- #
    parser.add_argument("--delimiter", default=",", help="Delimiter for data.")
    parser.add_argument("--col_names", nargs="+", default=["seqs", "labels"], help="Column names.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--trim", action="store_true", default=False, help="Whether to trim sequences (default: False).")

    # ----------------- BaseModelArguments ----------------- #
    parser.add_argument("--model_names", nargs="+", default=["ESM2-8"], help="List of model names to use.")

    # ----------------- ProbeArguments ----------------- #
    parser.add_argument("--probe_type", choices=["linear", "transformer", "crossconv"], default="linear", help="Type of probe.")
    parser.add_argument("--tokenwise", action="store_true", default=False, help="Tokenwise probe (default: False).")
    parser.add_argument("--hidden_dim", type=int, default=8192, help="Hidden dimension size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--pre_ln", action="store_false", default=True,
                        help="Disable pre-layernorm (default: enabled). Use --pre_ln to toggle off.")
    parser.add_argument("--classifier_dim", type=int, default=4096, help="Feed-forward dimension.")
    parser.add_argument("--transformer_dropout", type=float, default=0.1, help="Dropout rate for the transformer layers.")
    parser.add_argument("--classifier_dropout", type=float, default=0.2, help="Dropout rate for the classifier.")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads in multi-head attention.")
    parser.add_argument("--rotary", action="store_false", default=True,
                        help="Disable rotary embeddings (default: enabled). Use --rotary to toggle off.")
    parser.add_argument("--probe_pooling_types", nargs="+", default=["cls"], help="Pooling types to use.")

    # ----------------- EmbeddingArguments ----------------- #
    parser.add_argument("--embedding_batch_size", type=int, default=4, help="Batch size for embedding generation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument("--download_embeddings", action="store_true", default=False, help="Whether to download embeddings (default: False).")
    parser.add_argument("--matrix_embed", action="store_true", default=False, help="Use matrix embedding (default: False).")
    parser.add_argument("--embedding_pooling_types", nargs="+", default=["mean"], help="Pooling types for embeddings.")
    parser.add_argument("--save_embeddings", action="store_true", default=False, help="Save computed embeddings (default: False).")
    parser.add_argument("--embed_dtype", default="float16", help="Data type for embeddings.")
    parser.add_argument("--sql", action="store_true", default=False, help="Whether to use SQL storage (default: False).")

    # ----------------- TrainerArguments ----------------- #
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--trainer_batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")

    args = parser.parse_args()

    if args.hf_token is not None:
        from huggingface_hub import login
        login(args.hf_token)
    if args.wandb_api_key is not None:
        print('Wandb not integrated yet')
    if args.synthyra_api_key is not None:
        print('Synthyra API not integrated yet')

    if args.yaml_path is not None:
        with open(args.yaml_path, 'r') as file: 
            settings = yaml.safe_load(file)
        args = SimpleNamespace(**settings)
        return args
    else:
        return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.replay_path is not None:
        from logger import LogReplayer
        replayer = LogReplayer(args.replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = args.replay_path
        main = MainProcess(replay_args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        replayer.run_replay(main)
    else:
        main = MainProcess(args, GUI=False)
        for k, v in main.full_args.__dict__.items():
            print(f"{k}:\t{v}")
        main.apply_current_settings()
        main.get_datasets()
        print(len(main.all_seqs))
        main.save_embeddings_to_disk()
        main.run_probes()
        main.write_results()
    main.end_log()
