import torch
import tkinter as tk
from tkinter import ttk

from base_models.get_base_models import BaseModelArguments, standard_benchmark
from data.hf_data import HFDataArguments, get_hf_data
from data.supported_datasets import supported_datasets, possible_with_vector_reps
from embedder import EmbeddingArguments

from main import MainProcess


class GUI(MainProcess):
    def __init__(self, master):
        super().__init__()
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
        self.data_tab = ttk.Frame(self.notebook)
        self.embed_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.embed_tab, text="Embedding")

        # Build each tab
        self._build_model_tab()
        self._build_data_tab()
        self._build_embed_tab()

        #apply_button = ttk.Button(master, text="Stop code", command=self._clear_console)
        #apply_button.pack(side="bottom", pady=10)

    def _build_data_tab(self):
        # Label + Listbox for dataset names
        ttk.Label(self.data_tab, text="Dataset Names:").grid(row=0, column=0, padx=10, pady=5, sticky="nw")
        self.data_listbox = tk.Listbox(self.data_tab, selectmode="extended", height=25, width=25)
        for dataset_name in supported_datasets:
            self.data_listbox.insert(tk.END, dataset_name)
        self.data_listbox.grid(row=0, column=1, padx=10, pady=5, sticky="nw")

        # Max length (Spinbox)
        ttk.Label(self.data_tab, text="Max Sequence Length:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.settings_vars["max_len"] = tk.IntVar(value=1024)
        spin_max_len = ttk.Spinbox(
            self.data_tab,
            from_=1,
            to=8192,
            textvariable=self.settings_vars["max_len"]
        )
        spin_max_len.grid(row=1, column=1, padx=10, pady=5, sticky="w")

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

    def _get_data(self):
        print("=== Getting Data ===")

        # Gather selected indices from the listbox
        selected_indices = self.data_listbox.curselection()
        selected_datasets = [self.data_listbox.get(i) for i in selected_indices]

        # Optional: if nothing is selected, use all
        if not selected_datasets:
            selected_datasets = possible_with_vector_reps

        data_paths = [supported_datasets[name] for name in selected_datasets]
        self.data_args = HFDataArguments(
            data_paths=data_paths,
            max_len=self.settings_vars["max_len"].get(),
            trim=self.settings_vars["trim"].get()
        )
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

        # Convert embed_dtype string to actual torch.dtype if desired
        dtype_str = self.settings_vars["embed_dtype"].get()
        if dtype_str == "float32":
            dtype_val = torch.float32
        elif dtype_str == "float16":
            dtype_val = torch.float16
        elif dtype_str == "bfloat16":
            dtype_val = torch.bfloat16
        elif dtype_str == "float64":
            dtype_val = torch.float64
        else:
            dtype_val = torch.float32  # fallback

        self.embedding_args = EmbeddingArguments(
            all_seqs=self.all_seqs,
            batch_size=self.settings_vars["batch_size"].get(),
            num_workers=self.settings_vars["num_workers"].get(),
            download_embeddings=self.settings_vars["download_embeddings"].get(),
            matrix_embed=self.settings_vars["matrix_embed"].get(),
            pooling_types=pooling_list,
            save_embeddings=True,
            embed_dtype=dtype_val,
            sql=self.settings_vars["sql"].get(),
            save_dir=self.settings_vars["save_dir"].get()
        )

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

        self.model_args = BaseModelArguments(model_names=selected_models)

        # Do something with model_args here
        print("Models selected:")
        print(self.model_args)
        print("=========================\n")


def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
