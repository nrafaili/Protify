import torch
import tkinter as tk
import argparse
from tkinter import ttk
from base_models.get_base_models import BaseModelArguments, standard_benchmark
from data.hf_data import HFDataArguments
from data.supported_datasets import supported_datasets, possible_with_vector_reps
from embedder import EmbeddingArguments
from probes.get_probe import ProbeArguments
from main import MainProcess


class GUI(MainProcess):
    def __init__(self, master):
        super().__init__(argparse.Namespace())  # Initialize MainProcess with empty namespace
        self.master = master
        self.master.title("Settings GUI")
        self.master.geometry("600x800")

        # Store all settings in the full_args namespace
        self.full_args = self.full_args  # Reference from MainProcess parent

        icon = tk.PhotoImage(file="synthyra_logo.png")  
        # Set the window icon
        self.master.iconphoto(True, icon)

        # Dictionary to store Tkinter variables for settings
        self.settings_vars = {}

        # Create the Notebook widget
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each settings tab
        self.id_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.embed_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.probe_tab = ttk.Frame(self.notebook)
        self.trainer_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.id_tab, text="ID")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.embed_tab, text="Embedding")
        self.notebook.add(self.probe_tab, text="Probe")
        self.notebook.add(self.trainer_tab, text="Trainer")

        # Build each tab
        self._build_id_tab()
        self._build_model_tab()
        self._build_data_tab()
        self._build_embed_tab()
        self._build_probe_tab()
        self._build_trainer_tab()

        #apply_button = ttk.Button(master, text="Stop code", command=self._clear_console)
        #apply_button.pack(side="bottom", pady=10)

    def _build_id_tab(self):
        # Huggingface Username
        ttk.Label(self.id_tab, text="Huggingface Username:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_username"] = tk.StringVar(value="")
        entry_huggingface_username = ttk.Entry(self.id_tab, textvariable=self.settings_vars["huggingface_username"], width=20)
        entry_huggingface_username.grid(row=0, column=1, padx=10, pady=5)

        # Huggingface token
        ttk.Label(self.id_tab, text="Huggingface Token:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_token"] = tk.StringVar(value="")
        entry_huggingface_token = ttk.Entry(self.id_tab, textvariable=self.settings_vars["huggingface_token"], width=20)
        entry_huggingface_token.grid(row=1, column=1, padx=10, pady=5)

        # Synthyra API key
        ttk.Label(self.id_tab, text="Synthyra API Key:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["synthyra_api_key"] = tk.StringVar(value="")
        entry_synthyra_api_key = ttk.Entry(self.id_tab, textvariable=self.settings_vars["synthyra_api_key"], width=20)
        entry_synthyra_api_key.grid(row=2, column=1, padx=10, pady=5)

    def _build_data_tab(self):
        # Label + Listbox for dataset names
        ttk.Label(self.data_tab, text="Dataset Names:").grid(row=0, column=0, padx=10, pady=5, sticky="nw")
        self.data_listbox = tk.Listbox(self.data_tab, selectmode="extended", height=25, width=25)
        for dataset_name in supported_datasets:
            self.data_listbox.insert(tk.END, dataset_name)
        self.data_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")

        # Max length (Spinbox)
        ttk.Label(self.data_tab, text="Max Sequence Length:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["max_length"] = tk.IntVar(value=1024)
        spin_max_length = ttk.Spinbox(
            self.data_tab,
            from_=1,
            to=8192,
            textvariable=self.settings_vars["max_length"]
        )
        spin_max_length.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Trim (Checkbox)
        ttk.Label(self.data_tab, text="Trim Sequences:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["trim"] = tk.BooleanVar(value=False)
        check_trim = ttk.Checkbutton(
            self.data_tab,
            variable=self.settings_vars["trim"]
        )
        check_trim.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        run_button = ttk.Button(self.data_tab, text="Get Data", command=self._get_data)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _build_embed_tab(self):
        # batch_size
        ttk.Label(self.embed_tab, text="Batch Size:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["batch_size"] = tk.IntVar(value=4)
        spin_batch_size = ttk.Spinbox(self.embed_tab, from_=1, to=1024, textvariable=self.settings_vars["batch_size"])
        spin_batch_size.grid(row=1, column=1, padx=10, pady=5)

        # num_workers
        ttk.Label(self.embed_tab, text="Num Workers:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_workers"] = tk.IntVar(value=0)
        spin_num_workers = ttk.Spinbox(self.embed_tab, from_=0, to=64, textvariable=self.settings_vars["num_workers"])
        spin_num_workers.grid(row=2, column=1, padx=10, pady=5)

        # download_embeddings
        ttk.Label(self.embed_tab, text="Download Embeddings:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["download_embeddings"] = tk.BooleanVar(value=False)
        check_download = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["download_embeddings"])
        check_download.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # matrix_embed
        ttk.Label(self.embed_tab, text="Matrix Embedding:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["matrix_embed"] = tk.BooleanVar(value=False)
        check_matrix = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["matrix_embed"])
        check_matrix.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # pooling_types
        ttk.Label(self.embed_tab, text="Pooling Types (comma-separated):").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["pooling_types"] = tk.StringVar(value="mean")
        entry_pooling = ttk.Entry(self.embed_tab, textvariable=self.settings_vars["pooling_types"], width=20)
        entry_pooling.grid(row=5, column=1, padx=10, pady=5)

        # embed_dtype
        ttk.Label(self.embed_tab, text="Embedding DType:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embed_dtype"] = tk.StringVar(value="float32")
        combo_dtype = ttk.Combobox(
            self.embed_tab,
            textvariable=self.settings_vars["embed_dtype"],
            values=["float32", "float16", "bfloat16", "float64"]
        )
        combo_dtype.grid(row=7, column=1, padx=10, pady=5)

        # sql
        ttk.Label(self.embed_tab, text="Use SQL:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sql"] = tk.BooleanVar(value=False)
        check_sql = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["sql"])
        check_sql.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        # save_dir
        ttk.Label(self.embed_tab, text="Save Dir:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["save_dir"] = tk.StringVar(value="embeddings")
        entry_save_dir = ttk.Entry(self.embed_tab, textvariable=self.settings_vars["save_dir"], width=20)
        entry_save_dir.grid(row=9, column=1, padx=10, pady=5)

        run_button = ttk.Button(self.embed_tab, text="Embed sequences to disk", command=self._get_embeddings)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _build_model_tab(self):
        ttk.Label(self.model_tab, text="Model Names:").grid(row=0, column=0, padx=10, pady=5, sticky="nw")

        self.model_listbox = tk.Listbox(self.model_tab, selectmode="extended", height=10)
        for model_name in standard_benchmark:
            self.model_listbox.insert(tk.END, model_name)
        self.model_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")

        run_button = ttk.Button(self.model_tab, text="Select Models", command=self._select_models)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _build_probe_tab(self):
        # Probe Type
        ttk.Label(self.probe_tab, text="Probe Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_type"] = tk.StringVar(value="linear")
        combo_probe = ttk.Combobox(
            self.probe_tab,
            textvariable=self.settings_vars["probe_type"],
            values=["linear", "transformer", "crossconv"]
        )
        combo_probe.grid(row=0, column=1, padx=10, pady=5)

        # Tokenwise
        ttk.Label(self.probe_tab, text="Tokenwise:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["tokenwise"] = tk.BooleanVar(value=False)
        check_tokenwise = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["tokenwise"])
        check_tokenwise.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Pre Layer Norm
        ttk.Label(self.probe_tab, text="Pre Layer Norm:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["pre_ln"] = tk.BooleanVar(value=True)
        check_pre_ln = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["pre_ln"])
        check_pre_ln.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Number of Layers
        ttk.Label(self.probe_tab, text="Number of Layers:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_layers"] = tk.IntVar(value=1)
        spin_num_layers = ttk.Spinbox(self.probe_tab, from_=1, to=100, textvariable=self.settings_vars["num_layers"])
        spin_num_layers.grid(row=3, column=1, padx=10, pady=5)

        # Hidden Dimension
        ttk.Label(self.probe_tab, text="Hidden Dimension:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["hidden_dim"] = tk.IntVar(value=8192)
        spin_hidden_dim = ttk.Spinbox(self.probe_tab, from_=1, to=10000, textvariable=self.settings_vars["hidden_dim"])
        spin_hidden_dim.grid(row=4, column=1, padx=10, pady=5)

        # Dropout
        ttk.Label(self.probe_tab, text="Dropout:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["dropout"] = tk.DoubleVar(value=0.2)
        spin_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["dropout"])
        spin_dropout.grid(row=5, column=1, padx=10, pady=5)

        # Transformer Probe Settings
        ttk.Label(self.probe_tab, text="=== Transformer Probe Settings ===").grid(row=10, column=0, columnspan=2, pady=10)

        # FF Dimension
        ttk.Label(self.probe_tab, text="Classifier Dimension:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_dim"] = tk.IntVar(value=4096)
        spin_classifier_dim = ttk.Spinbox(self.probe_tab, from_=1, to=10000, textvariable=self.settings_vars["classifier_dim"])
        spin_classifier_dim.grid(row=11, column=1, padx=10, pady=5)

        # Classifier Dropout
        ttk.Label(self.probe_tab, text="Classifier Dropout:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_dropout"] = tk.DoubleVar(value=0.2)
        spin_class_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["classifier_dropout"])
        spin_class_dropout.grid(row=13, column=1, padx=10, pady=5)

        # Number of Heads
        ttk.Label(self.probe_tab, text="Number of Heads:").grid(row=14, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_heads"] = tk.IntVar(value=4)
        spin_num_heads = ttk.Spinbox(self.probe_tab, from_=1, to=32, textvariable=self.settings_vars["num_heads"])
        spin_num_heads.grid(row=14, column=1, padx=10, pady=5)

        # Rotary
        ttk.Label(self.probe_tab, text="Rotary:").grid(row=15, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["rotary"] = tk.BooleanVar(value=True)
        check_rotary = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["rotary"])
        check_rotary.grid(row=15, column=1, padx=10, pady=5, sticky="w")

        # Pooling Types
        ttk.Label(self.probe_tab, text="Pooling Types (comma-separated):").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_pooling_types"] = tk.StringVar(value="mean,cls")
        entry_pooling = ttk.Entry(self.probe_tab, textvariable=self.settings_vars["probe_pooling_types"], width=20)
        entry_pooling.grid(row=16, column=1, padx=10, pady=5)

        # Add a button to create the probe
        run_button = ttk.Button(self.probe_tab, text="Save Probe Arguments", command=self._create_probe_args)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _build_trainer_tab(self):
        # model_save_dir
        ttk.Label(self.trainer_tab, text="Model Save Dir:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["model_save_dir"] = tk.StringVar(value="probe")
        entry_model_save_dir = ttk.Entry(self.trainer_tab, textvariable=self.settings_vars["model_save_dir"], width=20)
        entry_model_save_dir.grid(row=0, column=1, padx=10, pady=5)

        # num_epochs
        ttk.Label(self.trainer_tab, text="Number of Epochs:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_epochs"] = tk.IntVar(value=200)
        spin_num_epochs = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["num_epochs"])
        spin_num_epochs.grid(row=1, column=1, padx=10, pady=5)

        # trainer_batch_size
        ttk.Label(self.trainer_tab, text="Trainer Batch Size:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["trainer_batch_size"] = tk.IntVar(value=64)
        spin_trainer_batch_size = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["trainer_batch_size"])
        spin_trainer_batch_size.grid(row=2, column=1, padx=10, pady=5)

        # gradient_accumulation_steps
        ttk.Label(self.trainer_tab, text="Gradient Accumulation Steps:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["gradient_accumulation_steps"] = tk.IntVar(value=1)
        spin_gradient_accumulation_steps = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["gradient_accumulation_steps"])
        spin_gradient_accumulation_steps.grid(row=3, column=1, padx=10, pady=5)

        # lr
        ttk.Label(self.trainer_tab, text="Learning Rate:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lr"] = tk.DoubleVar(value=1e-4)
        spin_lr = ttk.Spinbox(self.trainer_tab, from_=1e-6, to=1e-2, increment=1e-5, textvariable=self.settings_vars["lr"])
        spin_lr.grid(row=4, column=1, padx=10, pady=5)

        # patience
        ttk.Label(self.trainer_tab, text="Patience:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["patience"] = tk.IntVar(value=3)
        spin_patience = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["patience"])
        spin_patience.grid(row=5, column=1, padx=10, pady=5)

    def _create_probe_args(self):
        print("=== Creating Probe ===")
        
        # Convert pooling types string to list
        pooling_types = [p.strip() for p in self.settings_vars["probe_pooling_types"].get().split(",")]
        
        # Update full_args with probe settings
        self.full_args.probe_type = self.settings_vars["probe_type"].get()
        self.full_args.tokenwise = self.settings_vars["tokenwise"].get()
        self.full_args.hidden_dim = self.settings_vars["hidden_dim"].get()
        self.full_args.dropout = self.settings_vars["dropout"].get()
        self.full_args.num_layers = self.settings_vars["num_layers"].get()
        self.full_args.pre_ln = self.settings_vars["pre_ln"].get()
        self.full_args.classifier_dim = self.settings_vars["classifier_dim"].get()
        self.full_args.transformer_dropout = self.settings_vars["dropout"].get()
        self.full_args.classifier_dropout = self.settings_vars["classifier_dropout"].get()
        self.full_args.num_heads = self.settings_vars["num_heads"].get()
        self.full_args.rotary = self.settings_vars["rotary"].get()
        self.full_args.pooling_types = pooling_types

        # Create probe args from full args
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        
        print("Probe Arguments:")
        print(self.probe_args)
        print("========================\n")

    def _get_data(self):
        print("=== Getting Data ===")

        # Gather selected indices from the listbox
        selected_indices = self.data_listbox.curselection()
        selected_datasets = [self.data_listbox.get(i) for i in selected_indices]

        # Optional: if nothing is selected, use all
        if not selected_datasets:
            selected_datasets = possible_with_vector_reps

        data_paths = [supported_datasets[name] for name in selected_datasets]
        
        # Update full_args with data settings
        self.full_args.data_paths = data_paths
        self.full_args.max_length = self.settings_vars["max_length"].get()
        self.full_args.trim = self.settings_vars["trim"].get()

        # Create data args from full args
        self.data_args = HFDataArguments(**self.full_args.__dict__)
        
        print("Data Arguments:")
        print(self.data_args)
        print("========================\n")

        self.get_datasets()
        print("Data downloaded and stored")

    def _get_embeddings(self):
        if not self.all_seqs:
            print('Sequences are not loaded yet. Please run the data tab first.')
            return

        print("=== Embedding Sequences Code ===")

        # Convert comma-separated pooling_types into a list
        pooling_str = self.settings_vars["pooling_types"].get().strip()
        pooling_list = [p.strip() for p in pooling_str.split(",") if p.strip()]

        # Convert embed_dtype string to actual torch.dtype
        dtype_str = self.settings_vars["embed_dtype"].get()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64
        }
        dtype_val = dtype_map.get(dtype_str, torch.float32)

        # Update full_args with embedding settings
        self.full_args.all_seqs = self.all_seqs
        self.full_args.batch_size = self.settings_vars["batch_size"].get()
        self.full_args.num_workers = self.settings_vars["num_workers"].get()
        self.full_args.download_embeddings = self.settings_vars["download_embeddings"].get()
        self.full_args.matrix_embed = self.settings_vars["matrix_embed"].get()
        self.full_args.pooling_types = pooling_list
        self.full_args.save_embeddings = True
        self.full_args.embed_dtype = dtype_val
        self.full_args.sql = self.settings_vars["sql"].get()
        self.full_args.save_dir = self.settings_vars["save_dir"].get()

        # Create embedding args from full args
        self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)

        print("Saving embeddings to disk")
        self.save_embeddings_to_disk()
        print("Embeddings saved to disk")

    def _select_models(self):
        # Gather selected model names
        selected_indices = self.model_listbox.curselection()
        selected_models = [self.model_listbox.get(i) for i in selected_indices]

        # If no selection, default to the entire standard_benchmark
        if not selected_models:
            selected_models = standard_benchmark

        # Update full_args with model settings
        self.full_args.model_names = selected_models
        print(self.full_args.model_names)
        # Create model args from full args
        self.model_args = BaseModelArguments(**self.full_args.__dict__)

        print("Models selected:")
        print(self.model_args.__dict__)
        print("=========================\n")


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
