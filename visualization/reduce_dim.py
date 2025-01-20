import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE as SklearnTSNE
from typing import Optional, Union, List
from safetensors.torch import safe_open
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


@dataclass
class VisualizationArguments:
    embedding_save_dir: str = "embeddings"
    model_name: str = "ESM2-8"
    matrix_embed: bool = False
    sql: bool = False
    n_components: int = 2
    perplexity: float = 30.0  # for t-SNE
    n_neighbors: int = 15     # for UMAP
    min_dist: float = 0.1     # for UMAP
    seed: int = 42
    fig_size: tuple = (10, 10)
    save_fig: bool = True
    fig_dir: str = "figures"
    task_type: str = "singlelabel" # singlelabel, multilabel, regression


class DimensionalityReducer:
    """Base class for dimensionality reduction techniques"""
    def __init__(self, args: VisualizationArguments):
        self.args = args
        self.embeddings = None
        self.labels = None
        
    def load_embeddings(self, sequences: List[str], labels: Optional[List[Union[int, float]]] = None):
        """Load embeddings from file"""
        if self.args.sql:
            import sqlite3
            save_path = os.path.join(self.args.embedding_save_dir, 
                                   f'{self.args.model_name}_{self.args.matrix_embed}.db')
            embeddings = []
            with sqlite3.connect(save_path) as conn:
                c = conn.cursor()
                for seq in sequences:
                    c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
                    embedding = c.fetchone()[0]
                    embedding = np.frombuffer(embedding, dtype=np.float32)
                    embeddings.append(embedding)
        else:
            save_path = os.path.join(self.args.embedding_save_dir,
                                   f'{self.args.model_name}_{self.args.matrix_embed}.safetensors')
            embeddings = []
            with safe_open(save_path, framework="pt", device="cpu") as f:
                for seq in sequences:
                    embedding = f.get_tensor(seq).clone().numpy()
                    if self.args.matrix_embed:
                        # Take mean across sequence length for matrix embeddings
                        embedding = embedding.mean(axis=0)
                    embeddings.append(embedding)
                    
        self.embeddings = np.stack(embeddings)
        self.labels = np.array(labels) if labels is not None else None
        
    def fit_transform(self):
        """Implement in child class"""
        raise NotImplementedError
        
    def plot(self, save_name: Optional[str] = None):
        """Plot the reduced dimensionality embeddings with appropriate coloring scheme"""
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")
            
        reduced = self.fit_transform()
        plt.figure(figsize=self.args.fig_size)
        
        if self.labels is None:
            # No labels - use single color
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
            
        elif self.args.task_type == "singlelabel":
            unique_labels = np.unique(self.labels)
            if len(unique_labels) == 2:  # Binary classification
                colors = ['#ff7f0e', '#1f77b4']  # Orange and Blue
                cmap = LinearSegmentedColormap.from_list('binary', colors)
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                   c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=[0, 1])
                
            else:  # Multiclass classification
                # Use qualitative colormap for discrete classes
                n_classes = len(unique_labels)
                if n_classes <= 10:
                    cmap = 'tab10'
                elif n_classes <= 20:
                    cmap = 'tab20'
                else:
                    # For many classes, create custom colormap
                    colors = sns.color_palette('husl', n_colors=n_classes)
                    cmap = LinearSegmentedColormap.from_list('custom', colors)
                
                scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                                   c=self.labels, cmap=cmap, alpha=0.6)
                plt.colorbar(scatter, ticks=unique_labels)
                
        elif self.args.task_type == "multilabel":
            # For multilabel, color by number of positive labels
            label_counts = np.sum(self.labels, axis=1)
            viridis = plt.cm.viridis
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                               c=label_counts, cmap=viridis, alpha=0.6)
            plt.colorbar(scatter, label='Number of labels')
            
        elif self.args.task_type == "regression":
            # For regression, use sequential colormap
            vmin, vmax = np.percentile(self.labels, [2, 98])  # Robust scaling
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], 
                               c=self.labels, cmap='viridis', 
                               norm=norm, alpha=0.6)
            plt.colorbar(scatter, label='Value')
        
        plt.title(f'{self.__class__.__name__} visualization of {self.args.model_name} embeddings')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        if save_name is not None and self.args.save_fig:
            os.makedirs(self.args.fig_dir, exist_ok=True)
            plt.savefig(os.path.join(self.args.fig_dir, save_name), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()


class PCA(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.pca = SklearnPCA(n_components=args.n_components, random_state=args.seed)
        
    def fit_transform(self):
        return self.pca.fit_transform(self.embeddings)


class TSNE(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.tsne = SklearnTSNE(
            n_components=args.n_components,
            perplexity=args.perplexity,
            random_state=args.seed
        )
        
    def fit_transform(self):
        return self.tsne.fit_transform(self.embeddings)


class UMAP(DimensionalityReducer):
    def __init__(self, args: VisualizationArguments):
        super().__init__(args)
        self.umap = umap.UMAP(
            n_components=args.n_components,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed
        )
        
    def fit_transform(self):
        return self.umap.fit_transform(self.embeddings)


if __name__ == "__main__":
    # Example usage
    from data.hf_data import HFDataArguments, get_hf_data
    
    # Get some example data
    data_args = HFDataArguments(data_paths=["EC"])
    datasets, all_seqs = get_hf_data(data_args)
    
    # Get sequences and labels from first dataset
    dataset_name = list(datasets.keys())[0]
    train_set = datasets[dataset_name][0]
    sequences = train_set["seqs"]
    labels = train_set["labels"]
    
    # Set up visualization arguments
    vis_args = VisualizationArguments(
        embedding_save_dir="embeddings",
        model_name="ESM2-8",
        matrix_embed=False,
        sql=False
    )
    
    # Try each visualization technique
    for Reducer in [PCA, TSNE, UMAP]:
        reducer = Reducer(vis_args)
        reducer.load_embeddings(sequences, labels)
        reducer.plot(f"{dataset_name}_{Reducer.__name__}.png") 