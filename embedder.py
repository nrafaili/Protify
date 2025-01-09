import os
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class EmbeddingArguments:
    """
    """
    datasets: list[str]
    all_seqs: list[str]
    batch_size: int = 4
    num_workers: int = 0
    download_embeddings: bool = False
    max_length: int = 2048
    matrix_embed: bool = False
    pooling_types: list[str] = ['mean']
    save_embeddings: bool = False
    embed_dtype: torch.dtype = torch.float32
    sql: bool = False
    save_dir: str = 'embeddings'


class Pooler:
    def __init__(self, pooling_types: list[str]):
        self.pooling_types = pooling_types
        self.pooling_options = {
            'mean': self.mean_pooling,
            'max': self.max_pooling,
            'min': self.min_pooling,
            'norm': self.norm_pooling,
            'prod': self.prod_pooling,
            'median': self.median_pooling,
            'std': self.std_pooling,
            'var': self.var_pooling,
            'cls': self.cls_pooling,
        }

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def max_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.max(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).max(dim=1).values
    
    def min_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.min(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).min(dim=1).values

    def norm_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.norm(dim=1, p=2)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).norm(dim=1, p=2)

    def prod_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        length = emb.shape[1]
        if attention_mask is None:
            return emb.prod(dim=1) / length
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return ((emb * attention_mask).prod(dim=1) / attention_mask.sum(dim=1)) / length

    def median_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.median(dim=1).values
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).median(dim=1).values
    
    def std_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.std(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).std(dim=1)
    
    def var_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.var(dim=1)
        else:
            attention_mask = attention_mask.unsqueeze(-1)
            return (emb * attention_mask).var(dim=1)

    def cls_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        return emb[0, :]

    def __call__(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # [mean, max]
        final_emb = []
        for pooling_type in self.pooling_types:
            final_emb.append(self.pooling_options[pooling_type](emb, attention_mask).flatten())
        return torch.cat(final_emb, dim=0) # (n_pooling_types * d,)


### Dataset for Embedding
class ProteinDataset(Dataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching sequences."""
        return tokenizer(sequences, return_tensors="pt", padding='longest', pad_to_multiple_of=8)
    return _collate_fn


class Embedder:
    def __init__(self, args: EmbeddingArguments):
        self.args = args
        self.datasets = args.datasets
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.max_length = args.max_length
        self.matrix_embed = args.matrix_embed
        self.pooling_types = args.pooling_types
        self.save_embeddings = args.save_embeddings
        self.save_dtype = args.save_dtype
        self.sql = args.sql
        self.save_dir = args.save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device {self.device} found')

    def _download_embeddings(self):
        pass

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        import sqlite3
        sequences = []
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT sequence FROM embeddings")
            while True:
                row = c.fetchone()
                if row is None:
                    break
                sequences.append(row[0])
        return set(sequences)

    def _embed_sequences(self, model_name: str, embedding_model: any, tokenizer: any) -> Optional[dict[str, torch.Tensor]]:
        """
        TODO
        probably switch to sequence inputs to embedding_model, not tokenized
        """
        model = embedding_model.to(self.device).eval()
        dataset = ProteinDataset(self.all_seqs)
        collate_fn = build_collator(tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)
        device = self.device
        pooler = Pooler(self.pooling_types) if not self.matrix_embed else None

        def _get_embeddings(residue_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if self.matrix_embed:
                return residue_embeddings
            else:
                return pooler(residue_embeddings, attention_mask)

        if self.sql:
            import sqlite3
            save_path = os.path.join(self.save_dir, f'{model_name}.db')
            conn = sqlite3.connect(save_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(save_path)
            to_embed = [seq for seq in self.all_seqs if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * self.batch_size:(i + 1) * self.batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        residue_embeddings = model(input_ids, attention_mask)
                        embeddings = _get_embeddings(residue_embeddings, attention_mask)

                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            if self.matrix_embed:
                                emb = emb[mask.bool()]
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None
        
        from safetensors.torch import save_file, safe_open
        embeddings_dict = {}
        save_path = os.path.join(self.save_dir, f'{model_name}.safetensors')
        if os.path.exists(save_path):
            print(f"Loading embeddings from {save_path}")
            with safe_open(save_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    embeddings_dict[key] = f.get_tensor(key)
            print(f"Loaded {len(embeddings_dict)} embeddings from {save_path}")
            to_embed = [seq for seq in self.all_seqs if seq not in embeddings_dict]
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            print(f"No embeddings found in {save_path}")
            to_embed = self.all_seqs
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * self.batch_size:(i + 1) * self.batch_size]
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                residue_embeddings = model(input_ids, attention_mask)
                residue_embeddings = residue_embeddings.to(self.embed_dtype)
                embeddings = _get_embeddings(residue_embeddings, attention_mask).cpu()
                for seq, emb in zip(seqs, embeddings):
                    embeddings_dict[seq] = emb
        
        if self.save_embeddings:
            print(f"Saving embeddings to {save_path}")
            save_file(embeddings_dict, save_path)
        return embeddings_dict

    def __call__(self, model_name: str, embedding_model: Optional[any] = None, tokenizer: Optional[any] = None):
        if self.download_embeddings:
            self._download_embeddings(model_name)
        else:
            assert embedding_model is not None and tokenizer is not None, "embedding_model and tokenizer must be provided if you are embedding on device"

            if self.device == 'cpu':
                warnings.warn("Downloading embeddings is recommended for CPU usage - Embedding on CPU will be extremely slow!")
            embedding_dict = self._embed_sequences(model_name, embedding_model, tokenizer)
            return embedding_dict


if __name__ == '__main__':
    """
    TODO
    Embed all supported datasets with all supported models
    """
    pass
