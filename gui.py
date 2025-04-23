import torch
import tkinter as tk
import argparse
import queue
import traceback
from types import SimpleNamespace
from tkinter import ttk
from base_models.get_base_models import BaseModelArguments, standard_benchmark
from data.supported_datasets import supported_datasets, standard_data_benchmark, internal_synthyra_datasets
from embedder import EmbeddingArguments
from probes.get_probe import ProbeArguments
from probes.trainers import TrainerArguments
from main import MainProcess
from concurrent.futures import ThreadPoolExecutor
from data.data_mixin import DataArguments
from probes.scikit_classes import ScikitArguments


class BackgroundTask:
    def __init__(self, target, *args, **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.error = None
        self._complete = False
        
    def run(self):
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except Exception as e:
            self.error = e
            print(f"Error in background task: {str(e)}")
            traceback.print_exc()
        finally:
            self._complete = True
    
    @property
    def complete(self):
        return self._complete


class GUI(MainProcess):
    def __init__(self, master):
        super().__init__(argparse.Namespace(), GUI=True)  # Initialize MainProcess with empty namespace
        self.master = master
        self.master.title("Settings GUI")
        self.master.geometry("600x800")

        icon = tk.PhotoImage(file="synthyra_logo.png")  
        # Set the window icon
        self.master.iconphoto(True, icon)

        # Dictionary to store Tkinter variables for settings
        self.settings_vars = {}

        # Create the Notebook widget
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Create frames for each settings tab
        self.info_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.embed_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.probe_tab = ttk.Frame(self.notebook)
        self.trainer_tab = ttk.Frame(self.notebook)
        self.replay_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.info_tab, text="Info")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.embed_tab, text="Embedding")
        self.notebook.add(self.probe_tab, text="Probe")
        self.notebook.add(self.trainer_tab, text="Trainer")
        self.notebook.add(self.replay_tab, text="Replay")

        # Build these lines
        self.task_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.current_task = None
        
        # Start the queue checker
        self.check_task_queue()

        # Build each tab
        self.build_info_tab()
        self.build_model_tab()
        self.build_data_tab()
        self.build_embed_tab()
        self.build_probe_tab()
        self.build_trainer_tab()
        self.build_replay_tab()

    def check_task_queue(self):
        """Periodically check for completed background tasks"""
        if self.current_task and self.current_task.complete:
            if self.current_task.error:
                print(f"Task failed: {self.current_task.error}")
            self.current_task = None
            
        if not self.current_task and not self.task_queue.empty():
            self.current_task = self.task_queue.get()
            self.thread_pool.submit(self.current_task.run)
        
        # Schedule next check
        self.master.after(100, self.check_task_queue)
    
    def run_in_background(self, target, *args, **kwargs):
        """Queue a task to run in background"""
        task = BackgroundTask(target, *args, **kwargs)
        self.task_queue.put(task)
        return task

    def build_info_tab(self):
        # Create a frame for IDs
        id_frame = ttk.LabelFrame(self.info_tab, text="Identification")
        id_frame.pack(fill="x", padx=10, pady=5)

        # Huggingface Username
        ttk.Label(id_frame, text="Huggingface Username:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_username"] = tk.StringVar(value="Synthyra")
        entry_huggingface_username = ttk.Entry(id_frame, textvariable=self.settings_vars["huggingface_username"], width=30)
        entry_huggingface_username.grid(row=0, column=1, padx=10, pady=5)

        # Huggingface token
        ttk.Label(id_frame, text="Huggingface Token:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["huggingface_token"] = tk.StringVar(value="")
        entry_huggingface_token = ttk.Entry(id_frame, textvariable=self.settings_vars["huggingface_token"], width=30)
        entry_huggingface_token.grid(row=1, column=1, padx=10, pady=5)

        # Wandb API key 
        ttk.Label(id_frame, text="Wandb API Key:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["wandb_api_key"] = tk.StringVar(value="")
        entry_wandb_api_key = ttk.Entry(id_frame, textvariable=self.settings_vars["wandb_api_key"], width=30)
        entry_wandb_api_key.grid(row=2, column=1, padx=10, pady=5)

        # Synthyra API key
        ttk.Label(id_frame, text="Synthyra API Key:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["synthyra_api_key"] = tk.StringVar(value="")
        entry_synthyra_api_key = ttk.Entry(id_frame, textvariable=self.settings_vars["synthyra_api_key"], width=30)
        entry_synthyra_api_key.grid(row=3, column=1, padx=10, pady=5)

        # Create a frame for paths
        paths_frame = ttk.LabelFrame(self.info_tab, text="Paths")
        paths_frame.pack(fill="x", padx=10, pady=5)

        # Log directory
        ttk.Label(paths_frame, text="Log Directory:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["log_dir"] = tk.StringVar(value="logs")
        entry_log_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["log_dir"], width=30)
        entry_log_dir.grid(row=0, column=1, padx=10, pady=5)

        # Results directory
        ttk.Label(paths_frame, text="Results Directory:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["results_dir"] = tk.StringVar(value="results")
        entry_results_dir = ttk.Entry(paths_frame, textvariable=self.settings_vars["results_dir"], width=30)
        entry_results_dir.grid(row=1, column=1, padx=10, pady=5)

        # Model save directory
        ttk.Label(paths_frame, text="Model Save Directory:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["model_save_dir"] = tk.StringVar(value="weights")
        entry_model_save = ttk.Entry(paths_frame, textvariable=self.settings_vars["model_save_dir"], width=30)
        entry_model_save.grid(row=2, column=1, padx=10, pady=5)

        # Embedding save directory
        ttk.Label(paths_frame, text="Embedding Save Directory:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embedding_save_dir"] = tk.StringVar(value="embeddings")
        entry_embed_save = ttk.Entry(paths_frame, textvariable=self.settings_vars["embedding_save_dir"], width=30)
        entry_embed_save.grid(row=3, column=1, padx=10, pady=5)

        # Download directory
        ttk.Label(paths_frame, text="Download Directory:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["download_dir"] = tk.StringVar(value="Synthyra/mean_pooled_embeddings")
        entry_download = ttk.Entry(paths_frame, textvariable=self.settings_vars["download_dir"], width=30)
        entry_download.grid(row=4, column=1, padx=10, pady=5)

        # button to start logging
        start_logging_button = ttk.Button(self.info_tab, text="Start session", command=self._session_start)
        start_logging_button.pack(pady=10)

    def _session_start(self):
        # Update session variables
        hf_token = self.settings_vars["huggingface_token"].get()
        synthyra_api_key = self.settings_vars["synthyra_api_key"].get()
        wandb_api_key = self.settings_vars["wandb_api_key"].get()
        
        def background_login():
            if hf_token:
                from huggingface_hub import login
                login(hf_token)
            if wandb_api_key:
                print('Wandb not integrated yet')
            if synthyra_api_key:
                print('Synthyra API not integrated yet')
            
            self.full_args.hf_username = self.settings_vars["huggingface_username"].get()
            self.full_args.hf_token = hf_token
            self.full_args.synthyra_api_key = synthyra_api_key
            self.full_args.wandb_api_key = wandb_api_key
            self.full_args.log_dir = self.settings_vars["log_dir"].get()
            self.full_args.results_dir = self.settings_vars["results_dir"].get()
            self.full_args.model_save_dir = self.settings_vars["model_save_dir"].get()
            self.full_args.embedding_save_dir = self.settings_vars["embedding_save_dir"].get()
            self.full_args.download_dir = self.settings_vars["download_dir"].get()
            self.full_args.replay_path = None
            self.logger_args = SimpleNamespace(**self.full_args.__dict__)
            self.start_log_gui()
        
        self.run_in_background(background_login)

    def build_data_tab(self):
        # Max length (Spinbox)
        ttk.Label(self.data_tab, text="Max Sequence Length:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["max_length"] = tk.IntVar(value=1024)
        spin_max_length = ttk.Spinbox(
            self.data_tab,
            from_=1,
            to=32768,
            textvariable=self.settings_vars["max_length"]
        )
        spin_max_length.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Trim (Checkbox)
        ttk.Label(self.data_tab, text="Trim Sequences:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["trim"] = tk.BooleanVar(value=False)
        check_trim = ttk.Checkbutton(
            self.data_tab,
            variable=self.settings_vars["trim"]
        )
        check_trim.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        # Delimiter for data files
        ttk.Label(self.data_tab, text="Delimiter:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["delimiter"] = tk.StringVar(value=",")
        entry_delimiter = ttk.Entry(self.data_tab, textvariable=self.settings_vars["delimiter"], width=5)
        entry_delimiter.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # Column names for data files (comma-separated)
        ttk.Label(self.data_tab, text="Column Names (comma-separated):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["col_names"] = tk.StringVar(value="seqs,labels")
        entry_col_names = ttk.Entry(self.data_tab, textvariable=self.settings_vars["col_names"], width=20)
        entry_col_names.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        # Label + Listbox for dataset names
        ttk.Label(self.data_tab, text="Dataset Names:").grid(row=4, column=0, padx=10, pady=5, sticky="nw")
        self.data_listbox = tk.Listbox(self.data_tab, selectmode="extended", height=25, width=25)
        for dataset_name in supported_datasets:
            if dataset_name not in internal_synthyra_datasets:
                self.data_listbox.insert(tk.END, dataset_name)
        self.data_listbox.grid(row=4, column=1, padx=10, pady=5, sticky="nw")

        run_button = ttk.Button(self.data_tab, text="Get Data", command=self._get_data)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _get_data(self):
        print("=== Getting Data ===")
        
        # Gather settings
        selected_indices = self.data_listbox.curselection()
        selected_datasets = [self.data_listbox.get(i) for i in selected_indices]
        
        if not selected_datasets:
            selected_datasets = standard_data_benchmark
            
        def background_get_data():
            # Update full_args with data settings
            self.full_args.data_names = selected_datasets
            self.full_args.data_dirs = []
            self.full_args.max_length = self.settings_vars["max_length"].get()
            self.full_args.trim = self.settings_vars["trim"].get()
            self.full_args.delimiter = self.settings_vars["delimiter"].get()
            self.full_args.col_names = self.settings_vars["col_names"].get().split(",")
            
            # Create data args and get datasets
            self.data_args = DataArguments(**self.full_args.__dict__)
            args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
            self.logger_args = SimpleNamespace(**args_dict)
            self._write_args()
            self.get_datasets()
            print("Data downloaded and stored")
            
        self.run_in_background(background_get_data)

    def build_embed_tab(self):
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
        self.settings_vars["embedding_pooling_types"] = tk.StringVar(value="mean")
        entry_pooling = ttk.Entry(self.embed_tab, textvariable=self.settings_vars["embedding_pooling_types"], width=20)
        entry_pooling.grid(row=5, column=1, padx=10, pady=5)
        ttk.Label(self.embed_tab, text="Options: mean, max, min, norm, prod, median, std, var, cls, parti").grid(row=6, column=0, columnspan=2, padx=10, pady=2, sticky="w")

        # embed_dtype
        ttk.Label(self.embed_tab, text="Embedding DType:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["embed_dtype"] = tk.StringVar(value="float32")
        combo_dtype = ttk.Combobox(
            self.embed_tab,
            textvariable=self.settings_vars["embed_dtype"],
            values=["float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]
        )
        combo_dtype.grid(row=7, column=1, padx=10, pady=5)

        # sql
        ttk.Label(self.embed_tab, text="Use SQL:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["sql"] = tk.BooleanVar(value=False)
        check_sql = ttk.Checkbutton(self.embed_tab, variable=self.settings_vars["sql"])
        check_sql.grid(row=8, column=1, padx=10, pady=5, sticky="w")

        run_button = ttk.Button(self.embed_tab, text="Embed sequences to disk", command=self._get_embeddings)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _get_embeddings(self):
        if not self.all_seqs:
            print('Sequences are not loaded yet. Please run the data tab first.')
            return
            
        # Gather settings
        pooling_str = self.settings_vars["embedding_pooling_types"].get().strip()
        pooling_list = [p.strip() for p in pooling_str.split(",") if p.strip()]
        dtype_str = self.settings_vars["embed_dtype"].get()
        dtype_val = self.dtype_map.get(dtype_str, torch.float32)
        
        def background_get_embeddings():
            # Update full args
            self.full_args.all_seqs = self.all_seqs
            self.full_args.embedding_batch_size = self.settings_vars["batch_size"].get()
            self.full_args.embedding_num_workers = self.settings_vars["num_workers"].get()
            self.full_args.download_embeddings = self.settings_vars["download_embeddings"].get()
            self.full_args.matrix_embed = self.settings_vars["matrix_embed"].get()
            self.full_args.embedding_pooling_types = pooling_list
            self.full_args.save_embeddings = True
            self.full_args.embed_dtype = dtype_val
            self.full_args.sql = self.settings_vars["sql"].get()
            
            self.embedding_args = EmbeddingArguments(**self.full_args.__dict__)
            args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
            self.logger_args = SimpleNamespace(**args_dict)
            self._write_args()
            
            print("Saving embeddings to disk")
            self.save_embeddings_to_disk()
            print("Embeddings saved to disk")
            
        self.run_in_background(background_get_embeddings)

    def build_model_tab(self):
        ttk.Label(self.model_tab, text="Model Names:").grid(row=0, column=0, padx=10, pady=5, sticky="nw")

        self.model_listbox = tk.Listbox(self.model_tab, selectmode="extended", height=10)
        for model_name in standard_benchmark + ['experimental']:
            self.model_listbox.insert(tk.END, model_name)
        self.model_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")

        run_button = ttk.Button(self.model_tab, text="Select Models", command=self._select_models)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

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

        print("Model Args:")
        for k, v in self.model_args.__dict__.items():
            if k != 'model_names':
                print(f"{k}:\n{v}")
        print("=========================\n")
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()

    def build_probe_tab(self):
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
        self.settings_vars["n_layers"] = tk.IntVar(value=1)
        spin_n_layers = ttk.Spinbox(self.probe_tab, from_=1, to=100, textvariable=self.settings_vars["n_layers"])
        spin_n_layers.grid(row=3, column=1, padx=10, pady=5)

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
        ttk.Label(self.probe_tab, text="=== Transformer Probe Settings ===").grid(row=6, column=0, columnspan=2, pady=10)

        # FF Dimension
        ttk.Label(self.probe_tab, text="Classifier Dimension:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_dim"] = tk.IntVar(value=4096)
        spin_classifier_dim = ttk.Spinbox(self.probe_tab, from_=1, to=10000, textvariable=self.settings_vars["classifier_dim"])
        spin_classifier_dim.grid(row=7, column=1, padx=10, pady=5)

        # Classifier Dropout
        ttk.Label(self.probe_tab, text="Classifier Dropout:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["classifier_dropout"] = tk.DoubleVar(value=0.2)
        spin_class_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["classifier_dropout"])
        spin_class_dropout.grid(row=8, column=1, padx=10, pady=5)

        # Number of Heads
        ttk.Label(self.probe_tab, text="Number of Heads:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["n_heads"] = tk.IntVar(value=4)
        spin_n_heads = ttk.Spinbox(self.probe_tab, from_=1, to=32, textvariable=self.settings_vars["n_heads"])
        spin_n_heads.grid(row=9, column=1, padx=10, pady=5)

        # Rotary
        ttk.Label(self.probe_tab, text="Rotary:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["rotary"] = tk.BooleanVar(value=True)
        check_rotary = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["rotary"])
        check_rotary.grid(row=10, column=1, padx=10, pady=5, sticky="w")

        # Pooling Types
        ttk.Label(self.probe_tab, text="Pooling Types (comma-separated):").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["probe_pooling_types"] = tk.StringVar(value="mean, cls")
        entry_pooling = ttk.Entry(self.probe_tab, textvariable=self.settings_vars["probe_pooling_types"], width=20)
        entry_pooling.grid(row=11, column=1, padx=10, pady=5)
        # Transformer Dropout
        ttk.Label(self.probe_tab, text="Transformer Dropout:").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["transformer_dropout"] = tk.DoubleVar(value=0.1)
        spin_transformer_dropout = ttk.Spinbox(self.probe_tab, from_=0.0, to=1.0, increment=0.1, textvariable=self.settings_vars["transformer_dropout"])
        spin_transformer_dropout.grid(row=12, column=1, padx=10, pady=5, sticky="w")
        # Save Model
        ttk.Label(self.probe_tab, text="Save Model:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["save_model"] = tk.BooleanVar(value=False)
        check_save_model = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["save_model"])
        check_save_model.grid(row=13, column=1, padx=10, pady=5, sticky="w")
        # Production Model
        ttk.Label(self.probe_tab, text="Production Model:").grid(row=14, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["production_model"] = tk.BooleanVar(value=False)
        check_prod_model = ttk.Checkbutton(self.probe_tab, variable=self.settings_vars["production_model"])
        check_prod_model.grid(row=14, column=1, padx=10, pady=5, sticky="w")
        # Add a button to create the probe
        run_button = ttk.Button(self.probe_tab, text="Save Probe Arguments", command=self._create_probe_args)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _create_probe_args(self):
        print("=== Creating Probe ===")
        
        # Convert pooling types string to list
        probe_pooling_types = [p.strip() for p in self.settings_vars["probe_pooling_types"].get().split(",")]
        
        # Update full_args with probe settings
        self.full_args.probe_type = self.settings_vars["probe_type"].get()
        self.full_args.tokenwise = self.settings_vars["tokenwise"].get()
        self.full_args.hidden_dim = self.settings_vars["hidden_dim"].get()
        self.full_args.dropout = self.settings_vars["dropout"].get()
        self.full_args.n_layers = self.settings_vars["n_layers"].get()
        self.full_args.pre_ln = self.settings_vars["pre_ln"].get()
        self.full_args.classifier_dim = self.settings_vars["classifier_dim"].get()
        self.full_args.transformer_dropout = self.settings_vars["transformer_dropout"].get()
        self.full_args.classifier_dropout = self.settings_vars["classifier_dropout"].get()
        self.full_args.n_heads = self.settings_vars["n_heads"].get()
        self.full_args.rotary = self.settings_vars["rotary"].get()
        self.full_args.probe_pooling_types = probe_pooling_types
        self.full_args.save_model = self.settings_vars["save_model"].get()
        self.full_args.production_model = self.settings_vars["production_model"].get()

        # Create probe args from full args
        self.probe_args = ProbeArguments(**self.full_args.__dict__)
        
        print("Probe Arguments:")
        for k, v in self.probe_args.__dict__.items():
            if k != 'model_names':
                print(f"{k}:\n{v}")
        print("========================\n")
        args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
        self.logger_args = SimpleNamespace(**args_dict)
        self._write_args()

    def build_trainer_tab(self):
        # Lora checkbox
        ttk.Label(self.trainer_tab, text="Use LoRA:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_lora"] = tk.BooleanVar(value=False)
        check_lora = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["use_lora"])
        check_lora.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # LoRA r
        ttk.Label(self.trainer_tab, text="LoRA r:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_r"] = tk.IntVar(value=8)
        spin_lora_r = ttk.Spinbox(self.trainer_tab, from_=1, to=128, textvariable=self.settings_vars["lora_r"])
        spin_lora_r.grid(row=1, column=1, padx=10, pady=5)

        # LoRA alpha
        ttk.Label(self.trainer_tab, text="LoRA alpha:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_alpha"] = tk.DoubleVar(value=32.0)
        spin_lora_alpha = ttk.Spinbox(self.trainer_tab, from_=1.0, to=128.0, increment=1.0, textvariable=self.settings_vars["lora_alpha"])
        spin_lora_alpha.grid(row=2, column=1, padx=10, pady=5)

        # LoRA dropout
        ttk.Label(self.trainer_tab, text="LoRA dropout:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lora_dropout"] = tk.DoubleVar(value=0.01)
        spin_lora_dropout = ttk.Spinbox(self.trainer_tab, from_=0.0, to=0.5, increment=0.01, textvariable=self.settings_vars["lora_dropout"])
        spin_lora_dropout.grid(row=3, column=1, padx=10, pady=5)

        # Hybrid Probe checkbox
        ttk.Label(self.trainer_tab, text="Hybrid Probe:").grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["hybrid_probe"] = tk.BooleanVar(value=False)
        check_hybrid_probe = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["hybrid_probe"])
        check_hybrid_probe.grid(row=4, column=1, padx=10, pady=5, sticky="w")

        # Full finetuning checkbox
        ttk.Label(self.trainer_tab, text="Full Finetuning:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["full_finetuning"] = tk.BooleanVar(value=False)
        check_full_ft = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["full_finetuning"])
        check_full_ft.grid(row=5, column=1, padx=10, pady=5, sticky="w")

        # num_epochs
        ttk.Label(self.trainer_tab, text="Number of Epochs:").grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["num_epochs"] = tk.IntVar(value=200)
        spin_num_epochs = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["num_epochs"])
        spin_num_epochs.grid(row=6, column=1, padx=10, pady=5)

        # trainer_batch_size
        ttk.Label(self.trainer_tab, text="Trainer Batch Size:").grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["trainer_batch_size"] = tk.IntVar(value=64)
        spin_trainer_batch_size = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["trainer_batch_size"])
        spin_trainer_batch_size.grid(row=7, column=1, padx=10, pady=5)

        # gradient_accumulation_steps
        ttk.Label(self.trainer_tab, text="Gradient Accumulation Steps:").grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["gradient_accumulation_steps"] = tk.IntVar(value=1)
        spin_gradient_accumulation_steps = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["gradient_accumulation_steps"])
        spin_gradient_accumulation_steps.grid(row=8, column=1, padx=10, pady=5)

        # lr
        ttk.Label(self.trainer_tab, text="Learning Rate:").grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["lr"] = tk.DoubleVar(value=1e-4)
        spin_lr = ttk.Spinbox(self.trainer_tab, from_=1e-6, to=1e-2, increment=1e-5, textvariable=self.settings_vars["lr"])
        spin_lr.grid(row=9, column=1, padx=10, pady=5)

        # weight_decay
        ttk.Label(self.trainer_tab, text="Weight Decay:").grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["weight_decay"] = tk.DoubleVar(value=0.00)
        spin_weight_decay = ttk.Spinbox(self.trainer_tab, from_=0.0, to=1.0, increment=0.01, textvariable=self.settings_vars["weight_decay"])
        spin_weight_decay.grid(row=10, column=1, padx=10, pady=5)

        # patience
        ttk.Label(self.trainer_tab, text="Patience:").grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["patience"] = tk.IntVar(value=3)
        spin_patience = ttk.Spinbox(self.trainer_tab, from_=1, to=100, textvariable=self.settings_vars["patience"])
        spin_patience.grid(row=11, column=1, padx=10, pady=5, sticky="w")

        # Random Seed
        ttk.Label(self.trainer_tab, text="Random Seed:").grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["seed"] = tk.IntVar(value=42)
        spin_seed = ttk.Spinbox(self.trainer_tab, from_=0, to=10000, textvariable=self.settings_vars["seed"])
        spin_seed.grid(row=12, column=1, padx=10, pady=5, sticky="w")

        # Use Scikit
        ttk.Label(self.trainer_tab, text="Use Scikit:").grid(row=13, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["use_scikit"] = tk.BooleanVar(value=False)
        check_scikit = ttk.Checkbutton(self.trainer_tab, variable=self.settings_vars["use_scikit"])
        check_scikit.grid(row=13, column=1, padx=10, pady=5, sticky="w")

        # Scikit Iterations
        ttk.Label(self.trainer_tab, text="Scikit Iterations:").grid(row=14, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_n_iter"] = tk.IntVar(value=10)
        spin_scikit_n_iter = ttk.Spinbox(self.trainer_tab, from_=1, to=1000, textvariable=self.settings_vars["scikit_n_iter"])
        spin_scikit_n_iter.grid(row=14, column=1, padx=10, pady=5, sticky="w")

        # Scikit CV Folds
        ttk.Label(self.trainer_tab, text="Scikit CV Folds:").grid(row=15, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_cv"] = tk.IntVar(value=3)
        spin_scikit_cv = ttk.Spinbox(self.trainer_tab, from_=1, to=10, textvariable=self.settings_vars["scikit_cv"])
        spin_scikit_cv.grid(row=15, column=1, padx=10, pady=5, sticky="w")

        # Scikit Random State
        ttk.Label(self.trainer_tab, text="Scikit Random State:").grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_random_state"] = tk.IntVar(value=42)
        spin_scikit_rand = ttk.Spinbox(self.trainer_tab, from_=0, to=10000, textvariable=self.settings_vars["scikit_random_state"])
        spin_scikit_rand.grid(row=16, column=1, padx=10, pady=5, sticky="w")

        # Scikit Model Name
        ttk.Label(self.trainer_tab, text="Scikit Model Name (optional):").grid(row=17, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["scikit_model_name"] = tk.StringVar(value="")
        entry_scikit_name = ttk.Entry(self.trainer_tab, textvariable=self.settings_vars["scikit_model_name"], width=30)
        entry_scikit_name.grid(row=17, column=1, padx=10, pady=5, sticky="w")

        run_button = ttk.Button(self.trainer_tab, text="Run trainer", command=self._run_trainer)
        run_button.grid(row=99, column=0, columnspan=2, pady=(10, 10))

    def _run_trainer(self):
        # Gather settings
        self.full_args.use_lora = self.settings_vars["use_lora"].get()
        self.full_args.hybrid_probe = self.settings_vars["hybrid_probe"].get()
        self.full_args.full_finetuning = self.settings_vars["full_finetuning"].get()
        self.full_args.lora_r = self.settings_vars["lora_r"].get()
        self.full_args.lora_alpha = self.settings_vars["lora_alpha"].get()
        self.full_args.lora_dropout = self.settings_vars["lora_dropout"].get()
        self.full_args.num_epochs = self.settings_vars["num_epochs"].get()
        self.full_args.trainer_batch_size = self.settings_vars["trainer_batch_size"].get()
        self.full_args.gradient_accumulation_steps = self.settings_vars["gradient_accumulation_steps"].get()
        self.full_args.lr = self.settings_vars["lr"].get()
        self.full_args.weight_decay = self.settings_vars["weight_decay"].get()
        self.full_args.patience = self.settings_vars["patience"].get()
        self.full_args.seed = self.settings_vars["seed"].get()
        self.full_args.use_scikit = self.settings_vars["use_scikit"].get()
        self.full_args.scikit_n_iter = self.settings_vars["scikit_n_iter"].get()
        self.full_args.scikit_cv = self.settings_vars["scikit_cv"].get()
        self.full_args.scikit_random_state = self.settings_vars["scikit_random_state"].get()
        self.full_args.scikit_model_name = self.settings_vars["scikit_model_name"].get()

        def background_run_trainer():
            self.trainer_args = TrainerArguments(**self.full_args.__dict__)
            self.scikit_args = ScikitArguments(**self.full_args.__dict__)
            args_dict = {k: v for k, v in self.full_args.__dict__.items() if k != 'all_seqs' and 'token' not in k.lower() and 'api' not in k.lower()}
            self.logger_args = SimpleNamespace(**args_dict)
            self._write_args()
            
            if self.full_args.use_scikit:
                self.run_scikit_scheme()
            elif self.full_args.use_lora:
                self.init_lora_model()
                self.run_lora_model()
            elif self.full_args.full_finetuning:
                self.get_full_finetuning_model()
                self.run_full_finetuning_model()
            elif self.full_args.hybrid_probe:
                self.init_hybrid_probe()
            else:
                self.run_nn_probe()
            
        self.run_in_background(background_run_trainer)

    def build_replay_tab(self):
        # Create a frame for replay settings
        replay_frame = ttk.LabelFrame(self.replay_tab, text="Log Replay Settings")
        replay_frame.pack(fill="x", padx=10, pady=5)

        # Replay log path
        ttk.Label(replay_frame, text="Replay Log Path:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["replay_path"] = tk.StringVar(value="")
        entry_replay = ttk.Entry(replay_frame, textvariable=self.settings_vars["replay_path"], width=40)
        entry_replay.grid(row=0, column=1, padx=10, pady=5)

        # Browse button for selecting log file
        browse_button = ttk.Button(replay_frame, text="Browse", command=self._browse_replay_log)
        browse_button.grid(row=0, column=2, padx=5, pady=5)

        # Start replay button
        replay_button = ttk.Button(replay_frame, text="Start Replay", command=self._start_replay)
        replay_button.grid(row=1, column=0, columnspan=3, pady=20)

    def _browse_replay_log(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Replay Log",
            filetypes=(("Txt files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.settings_vars["replay_path"].set(filename)

    def _start_replay(self):
        replay_path = self.settings_vars["replay_path"].get()
        if not replay_path:
            print("Please select a replay log file first")
            return
        
        from logger import LogReplayer
        replayer = LogReplayer(replay_path)
        replay_args = replayer.parse_log()
        replay_args.replay_path = replay_path
        
        # Update GUI with replay settings
        for key, value in replay_args.__dict__.items():
            if key in self.settings_vars:
                self.settings_vars[key].set(value)
        
        print(f"Loaded settings from {replay_path}")
        replayer.run_replay(self)


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
