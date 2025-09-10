import os
import argparse
import numpy as np
import torch

from utils import print_message, torch_load


def load_embeddings_dict(path: str) -> dict[str, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    emb = torch_load(path)
    if not isinstance(emb, dict):
        raise ValueError(f"Embedding file at {path} is not a dict")
    return emb


def stack_by_sequence_order(emb_dict: dict[str, torch.Tensor], seqs: list[str]) -> np.ndarray:
    return np.stack([emb_dict[s].numpy() for s in seqs], axis=0)


def compare_arrays(a: np.ndarray, b: np.ndarray, name: str, atol: float = 1e-8, rtol: float = 1e-5):
    if a.shape != b.shape:
        print_message(f"{name}: Shape mismatch {a.shape} vs {b.shape}")
        return
    allclose = np.allclose(a, b, atol=atol, rtol=rtol)
    array_eq = np.array_equal(a, b)
    max_abs_diff = float(np.max(np.abs(a - b)))
    mean_abs_diff = float(np.mean(np.abs(a - b)))
    print_message(f"{name} -> np.allclose: {allclose} | np.array_equal: {array_eq} | max_abs_diff: {max_abs_diff:.6g} | mean_abs_diff: {mean_abs_diff:.6g}")


def compare_runs(
    run_dir_a: str,
    run_dir_b: str,
    model_a: str = 'Random',
    model_b: str = 'Random-Transformer',
    atol: float = 1e-8,
    rtol: float = 1e-5,
):
    path_a_model_a = os.path.join(run_dir_a, f'{model_a}_False.pth')
    path_b_model_a = os.path.join(run_dir_b, f'{model_a}_False.pth')
    path_a_model_b = os.path.join(run_dir_a, f'{model_b}_False.pth')
    path_b_model_b = os.path.join(run_dir_b, f'{model_b}_False.pth')

    print_message("Loading embeddings")
    a_model_a = load_embeddings_dict(path_a_model_a)
    b_model_a = load_embeddings_dict(path_b_model_a)
    a_model_b = load_embeddings_dict(path_a_model_b)
    b_model_b = load_embeddings_dict(path_b_model_b)

    # Align sequence order: use intersection and a consistent order
    seqs_a = set(a_model_a.keys())
    seqs_b = set(b_model_a.keys())
    common_seqs_a = sorted(list(seqs_a.intersection(seqs_b)), key=len, reverse=True)

    seqs_c = set(a_model_b.keys())
    seqs_d = set(b_model_b.keys())
    common_seqs_b = sorted(list(seqs_c.intersection(seqs_d)), key=len, reverse=True)

    print_message(f"{model_a}: comparing {len(common_seqs_a)} sequences")
    a_arr_a = stack_by_sequence_order(a_model_a, common_seqs_a)
    b_arr_a = stack_by_sequence_order(b_model_a, common_seqs_a)
    compare_arrays(a_arr_a, b_arr_a, f'{model_a} (RunA vs RunB)', atol=atol, rtol=rtol)

    print_message(f"{model_b}: comparing {len(common_seqs_b)} sequences")
    a_arr_b = stack_by_sequence_order(a_model_b, common_seqs_b)
    b_arr_b = stack_by_sequence_order(b_model_b, common_seqs_b)
    compare_arrays(a_arr_b, b_arr_b, f'{model_b} (RunA vs RunB)', atol=atol, rtol=rtol)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare embeddings across two runs using np.allclose and np.array_equal')
    parser.add_argument('--run_dir_a', type=str, required=True, help='Path to first run output dir (contains .pth files)')
    parser.add_argument('--run_dir_b', type=str, required=True, help='Path to second run output dir (contains .pth files)')
    parser.add_argument('--model_a', type=str, default='Random', help='First model to compare (default: Random)')
    parser.add_argument('--model_b', type=str, default='Random-Transformer', help='Second model to compare (default: Random-Transformer)')
    parser.add_argument('--atol', type=float, default=1e-8, help='Absolute tolerance for np.allclose')
    parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for np.allclose')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare_runs(
        run_dir_a=args.run_dir_a,
        run_dir_b=args.run_dir_b,
        model_a=args.model_a,
        model_b=args.model_b,
        atol=args.atol,
        rtol=args.rtol,
    )


