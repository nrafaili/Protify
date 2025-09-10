import os
import argparse
import torch

from ..seed_utils import set_global_seed, set_determinism
from ..data.data_mixin import DataArguments, DataMixin
from ..embedder import EmbeddingArguments, Embedder
from ..utils import print_message

def generate_embeddings(
    data_name: str,
    seed: int,
    deterministic: bool,
    run_tag: str,
):

    print_message(f"Loading dataset: {data_name}")
    data_args = DataArguments(
        data_names=data_name,
        max_length=1024,
        trim=False,
    )
    datasets, all_seqs = DataMixin(data_args).get_data()

    print_message(f"Preparing to embed {len(all_seqs)} sequences")

    out_dir = os.path.join('embeddings', f'{data_name}_seed{seed}_det{deterministic}_{run_tag}')
    os.makedirs(out_dir, exist_ok=True)

    emb_args = EmbeddingArguments(
        embedding_batch_size=batch_size,
        embedding_num_workers=0,
        download_embeddings=False,
        matrix_embed=False,
        embedding_pooling_types=['mean'],
        save_embeddings=True,
        embed_dtype=torch.float32,
        sql=False,
        embedding_save_dir=out_dir,
    )

    embedder = Embedder(emb_args, all_seqs)

    # Embed with Random-Transformer
    print_message("Embedding with Random-Transformer")
    _ = embedder('Random-Transformer')

    # Embed with Random
    print_message("Embedding with Random")
    _ = embedder('Random')

    print_message(f"Done. Saved to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate embeddings for a single run (Random and Random-Transformer)')
    parser.add_argument('--data_names', type=str, default='DeepLoc-2', help='Dataset short name (e.g., DeepLoc-2)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Enable deterministic algorithms')
    parser.add_argument('--run_tag', type=str, required=True, help='Run identifier, e.g., run1 or 2025-09-10-1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Set seed and determinism according to args
    set_global_seed(args.seed)
    if args.deterministic:
        set_determinism()
    print(f"Seed: {set_global_seed.__defaults__ if False else args.seed}, Deterministic: {args.deterministic}")
    generate_embeddings(
        data_name=args.data_names,
        seed=args.seed,
        deterministic=args.deterministic,
        run_tag=args.run_tag,
    )


