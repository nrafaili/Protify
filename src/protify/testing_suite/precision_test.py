"""
Precision Comparison Test Suite

Tests masked marginal scoring across FP32, FP16, and BF16 precision modes.
Compares logits and final masked marginal Δlog-prob scores across FP32, FP16, and BF16.

Alternatively, the `--embeddings_test` option supports general embedding precision
testing using SwissProt sequences.
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from dataclasses import dataclass
from datasets import load_dataset

from benchmarks.proteingym.scorer import ProteinGymScorer, SequenceProcessor
from benchmarks.proteingym.data_loader import load_proteingym_dms
from base_models.get_base_models import get_base_model
from seed_utils import set_global_seed, seed_worker, dataloader_generator, get_global_seed
from embedder import Embedder, EmbeddingArguments, build_collator
from pooler import Pooler
from data.dataset_classes import SimpleProteinDataset
from torch.utils.data import DataLoader


TEST_DMS_IDS = [
    "A4_HUMAN_Seuma_2022",  # Stability
    "ACE2_HUMAN_Chan_2020",  # Binding
    "ENV_HV1BR_Haddox_2016",  # Organismal fitness
]


def score_with_precision(
    model: Any,
    tokenizer: Any,
    model_name: str,
    device: torch.device,
    df: pd.DataFrame,
    target_seq: str,
    dtype: Optional[torch.dtype] = None,
    max_batch_tokens: int = 16384,
) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Score variants using ProteinGymScorer with specified precision.
    
    Creates a ProteinGymScorer instance with appropriate precision settings
    and uses _position_log_probs_batched with return_logits=True.
    
    Args:
        model: The model to run inference on
        tokenizer: The tokenizer
        model_name: Name of the model
        device: Device to run on
        df: DataFrame with variants to score
        target_seq: Target sequence
        dtype: Optional dtype for autocast (None for FP32)
        max_batch_tokens: Maximum tokens per batch
        
    Returns:
        Tuple of (scores list, logits list)
    """
    if dtype is None:
        use_autocast = False
        scorer_dtype = None
    else:
        use_autocast = True
        scorer_dtype = dtype
    
    # Create scorer with appropriate precision settings
    scorer = ProteinGymScorer(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=32,
        max_batch_tokens=max_batch_tokens,
        use_autocast=use_autocast,
        dtype=scorer_dtype,
    )
    
    # Build mutation info
    encoded_target = np.frombuffer(target_seq.encode(), dtype=np.uint8)
    mutation_info = {}
    for row in df.itertuples(index=False):
        mutant = row.mutant
        if mutant in mutation_info:
            continue
        mutated_seq = row.mutated_seq
        mismatches = np.array(
            SequenceProcessor.find_mismatches(encoded_target, mutated_seq),
            dtype=np.int64
        )
        wt_aas = ''.join(target_seq[p] for p in mismatches)
        mt_aas = ''.join(mutated_seq[p] for p in mismatches)
        mutation_info[mutant] = (mismatches, wt_aas, mt_aas)
    
    # Get window info (no slicing needed since all sequences < 1022)
    seq_len = len(target_seq)
    uniq_mutants = pd.unique(df["mutant"])
    window_info = {
        m: {"window_start": 0, "window_end": seq_len, "sliced_seq": target_seq}
        for m in uniq_mutants
    }
    
    # Group by position for efficient batching
    position_groups: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, np.ndarray, str, str]]] = {}
    
    for row_idx, row in enumerate(df.itertuples(index=False)):
        mutant = row.mutant
        positions, wt_aas, mt_aas = mutation_info[mutant]
        
        window = window_info.get(mutant)
        if window is None:
            raise ValueError(f"No available window for mutant {mutant}")
        
        window_start = window['window_start']
        window_end = window['window_end']
        
        min_pos = positions.min()
        max_pos = positions.max()
        if not (window_start <= min_pos and max_pos < window_end):
            raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
        
        pos_rel = positions - window_start
        pos_tuple = tuple(sorted(pos_rel))
        
        key = (window_start, window_end, pos_tuple)
        position_groups.setdefault(key, []).append((row_idx, positions, wt_aas, mt_aas))
    
    sequences: List[str] = []
    positions_list: List[List[int]] = []
    variant_info: List[List[Tuple[int, str, str]]] = []
    
    for (window_start, window_end, pos_tuple), variants in position_groups.items():
        window_seq = target_seq[window_start:window_end]
        sequences.append(window_seq)
        positions_list.append(list(pos_tuple))
        variant_info.append([(row_idx, wt_aas, mt_aas) for row_idx, _, wt_aas, mt_aas in variants])
    
    # Use scorer's _position_log_probs_batched with return_logits=True
    all_log_probs, all_logits = scorer._position_log_probs_batched(
        "masked_marginal",
        sequences,
        positions_list,
        return_logits=True,
    )
    
    # Calculate scores from log_probs
    scores = [0.0] * len(df)
    all_variant_logits = [None] * len(df)
    
    for group_idx, (variants_in_group, log_probs) in enumerate(zip(variant_info, all_log_probs)):
        num_variants = len(variants_in_group)
        num_positions = log_probs.size(0)
        
        wt_ids_list = []
        mt_ids_list = []
        
        for row_idx, wt_aas, mt_aas in variants_in_group:
            assert len(wt_aas) == num_positions, f"Variant {row_idx} in group has {len(wt_aas)} muts, expected {num_positions}"
            wt_ids = [scorer.aa_to_id[aa] for aa in wt_aas]
            mt_ids = [scorer.aa_to_id[aa] for aa in mt_aas]
            wt_ids_list.append(wt_ids)
            mt_ids_list.append(mt_ids)
        
        wt_tensor = torch.tensor(wt_ids_list, device=log_probs.device, dtype=torch.long)
        mt_tensor = torch.tensor(mt_ids_list, device=log_probs.device, dtype=torch.long)
        
        pos_idx = torch.arange(num_positions, device=log_probs.device)[None, :].expand(num_variants, -1)
        wt_log_probs = log_probs[pos_idx, wt_tensor]
        mt_log_probs = log_probs[pos_idx, mt_tensor]
        deltas = (mt_log_probs - wt_log_probs).sum(dim=1)
        
        group_logits = all_logits[group_idx]
        
        # Check for NaN/non-finite values in deltas
        if not torch.isfinite(deltas).all():
            nan_count = torch.isnan(deltas).sum().item()
            inf_count = torch.isinf(deltas).sum().item()
            raise ValueError(
                f"Non-finite values in score deltas for group {group_idx}: "
                f"{nan_count} NaNs, {inf_count} Infs"
            )
        
        for i, (row_idx, _, _) in enumerate(variants_in_group):
            scores[row_idx] = deltas[i].item()
            all_variant_logits[row_idx] = group_logits
    
    # Final validation of all scores
    scores_arr = np.array(scores)
    if not np.all(np.isfinite(scores_arr)):
        nan_count = np.isnan(scores_arr).sum()
        inf_count = np.isinf(scores_arr).sum()
        raise ValueError(
            f"Non-finite values in final scores: {nan_count} NaNs, {inf_count} Infs"
        )
    
    return scores, all_variant_logits


def compare_logits(fp32_logits: List[torch.Tensor], other_logits: List[torch.Tensor]) -> Tuple[float, float]:
    """
    Compare logits between FP32 and another precision mode.
    
    Args:
        fp32_logits: List of FP32 logit tensors
        other_logits: List of logit tensors from another precision
        
    Returns:
        Tuple of (MSE, Max Absolute Difference)
    """
    all_mse = []
    all_max_diff = []
    
    for idx, (fp32_log, other_log) in enumerate(zip(fp32_logits, other_logits)):
        if fp32_log is None or other_log is None:
            raise ValueError(f"Logits at index {idx} is None")
        
        # Check for NaN/non-finite values in both tensors
        if not torch.isfinite(fp32_log).all():
            nan_count = torch.isnan(fp32_log).sum().item()
            inf_count = torch.isinf(fp32_log).sum().item()
            raise ValueError(
                f"Non-finite values in FP32 logits at index {idx}: "
                f"{nan_count} NaNs, {inf_count} Infs"
            )
        if not torch.isfinite(other_log).all():
            nan_count = torch.isnan(other_log).sum().item()
            inf_count = torch.isinf(other_log).sum().item()
            raise ValueError(
                f"Non-finite values in other precision logits at index {idx}: "
                f"{nan_count} NaNs, {inf_count} Infs"
            )
        
        diff = fp32_log - other_log
        mse = (diff ** 2).mean().item()
        max_diff = diff.abs().max().item()
        
        # Verify computed values are finite
        if not np.isfinite(mse) or not np.isfinite(max_diff):
            raise ValueError(
                f"Non-finite comparison result at index {idx}: MSE={mse}, max_diff={max_diff}"
            )
        
        all_mse.append(mse)
        all_max_diff.append(max_diff)
    
    if len(all_mse) == 0:
        raise ValueError("No valid logits to compare")
    
    return np.mean(all_mse), np.max(all_max_diff)


def compare_scores(fp32_scores: List[float], other_scores: List[float]) -> float:
    """
    Compare scores using Spearman correlation.
    
    Args:
        fp32_scores: List of FP32 scores
        other_scores: List of scores from another precision
        
    Returns:
        Spearman correlation coefficient
    """
    if len(fp32_scores) != len(other_scores):
        raise ValueError(f"Score lists have different lengths: {len(fp32_scores)} vs {len(other_scores)}")
    
    fp32_arr = np.array(fp32_scores)
    other_arr = np.array(other_scores)
    
    if not np.all(np.isfinite(fp32_arr)) or not np.all(np.isfinite(other_arr)):
        raise ValueError("Non-finite values found in score arrays")
    
    corr, _ = spearmanr(fp32_arr, other_arr)
    
    if not np.isfinite(corr):
        raise ValueError(f"Non-finite Spearman correlation: {corr}")
    
    return corr


def plot_error_histogram(
    fp32_scores: List[float],
    fp16_scores: List[float],
    bf16_scores: List[float],
    output_path: str,
    model_name: str,
    num_assays: int,
):
    """
    Plot histogram of score errors (FP32 - FP16 and FP32 - BF16) averaged across assays.
    
    Args:
        fp32_scores: List of FP32 scores (concatenated from all assays)
        fp16_scores: List of FP16 scores (concatenated from all assays)
        bf16_scores: List of BF16 scores (concatenated from all assays)
        output_path: Path to save the plot
        model_name: Model name
        num_assays: Number of assays the scores were aggregated from
    """
    fp32_arr = np.array(fp32_scores)
    fp16_arr = np.array(fp16_scores)
    bf16_arr = np.array(bf16_scores)
    
    # Validate all score arrays are finite
    if not np.all(np.isfinite(fp32_arr)):
        nan_count = np.isnan(fp32_arr).sum()
        inf_count = np.isinf(fp32_arr).sum()
        raise ValueError(f"Non-finite values in FP32 scores: {nan_count} NaNs, {inf_count} Infs")
    if not np.all(np.isfinite(fp16_arr)):
        nan_count = np.isnan(fp16_arr).sum()
        inf_count = np.isinf(fp16_arr).sum()
        raise ValueError(f"Non-finite values in FP16 scores: {nan_count} NaNs, {inf_count} Infs")
    if not np.all(np.isfinite(bf16_arr)):
        nan_count = np.isnan(bf16_arr).sum()
        inf_count = np.isinf(bf16_arr).sum()
        raise ValueError(f"Non-finite values in BF16 scores: {nan_count} NaNs, {inf_count} Infs")
    
    fp16_errors = (fp32_arr - fp16_arr)
    bf16_errors = (fp32_arr - bf16_arr)
    
    plt.figure(figsize=(10, 6))
    plt.hist(fp16_errors, bins=50, edgecolor='black', alpha=0.6, label='FP16 Error (FP32 - FP16)', color='blue')
    plt.hist(bf16_errors, bins=50, edgecolor='black', alpha=0.6, label='BF16 Error (FP32 - BF16)', color='red')
    plt.xlabel('Score Error (FP32 - Precision)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of Score Errors (Averaged Across {num_assays} Assays)\n{model_name}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    fp16_mean = np.mean(fp16_errors)
    fp16_std = np.std(fp16_errors)
    bf16_mean = np.mean(bf16_errors)
    bf16_std = np.std(bf16_errors)
    stats_text = (f'FP16 - Mean: {fp16_mean:.6f}, Std: {fp16_std:.6f}, N: {len(fp16_errors)}\n'
                  f'BF16 - Mean: {bf16_mean:.6f}, Std: {bf16_std:.6f}, N: {len(bf16_errors)}')
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_precision_test(
    dms_ids: List[str],
    model_names: List[str],
    seed: Optional[int] = None,
    output_dir: str = "precision_test_results",
):
    """
    Run precision comparison test across models and DMS assays.
    """
    seed = set_global_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_models = list(ProteinGymScorer.MODEL_CONTEXT_LENGTH.keys())
    test_models = [m for m in all_models if not m.lower().startswith("e1")]
    
    if model_names:
        test_models = [m for m in test_models if m in model_names]
    
    print(f"Testing {len(test_models)} models: {test_models}")
    print(f"Testing {len(dms_ids)} DMS assays: {dms_ids}")
    
    results_summary = []
    
    for model_name in tqdm(test_models, desc="Models"):
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")
        
        # Collect scores across all assays for this model
        all_fp32_scores: List[float] = []
        all_fp16_scores: List[float] = []
        all_bf16_scores: List[float] = []
        assays_processed = 0
        
        try:
            model, tokenizer = get_base_model(model_name, masked_lm=True)
            model = model.to(device)
            model.eval()
            
            for dms_id in tqdm(dms_ids, desc=f"Assays ({model_name})", leave=False):
                print(f"\n  Processing DMS: {dms_id}")
                
                try:
                    df = load_proteingym_dms(dms_id, mode="benchmark", repo_id="GleghornLab/ProteinGym_DMS")
                    if df is None or len(df) == 0:
                        print(f"Warning: No data for {dms_id}")
                        continue
                    
                    target_seq = df['target_seq'].iloc[0]
                    seq_len = len(target_seq)
                    
                    # Verify sequence fits in context window
                    context_len = ProteinGymScorer.MODEL_CONTEXT_LENGTH.get(model_name, 1022)
                    if seq_len > context_len:
                        print(f"Warning: Sequence length {seq_len} > context length {context_len} for {model_name}")
                        continue
                    
                    # Run scoring with different precisions
                    # Clear GPU cache before each run to prevent OOM
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"    Running FP32...")
                    set_global_seed(seed)  # Reset seed for each precision run
                    fp32_scores, fp32_logits = score_with_precision(
                        model, tokenizer, model_name, device, df, target_seq, dtype=None
                    )
                    # Move logits to CPU immediately to free GPU memory
                    fp32_logits = [l.cpu() if l is not None else None for l in fp32_logits]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"    Running FP16...")
                    set_global_seed(seed)
                    fp16_scores, fp16_logits = score_with_precision(
                        model, tokenizer, model_name, device, df, target_seq, dtype=torch.float16
                    )
                    fp16_logits = [l.cpu() if l is not None else None for l in fp16_logits]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"    Running BF16...")
                    set_global_seed(seed)
                    bf16_scores, bf16_logits = score_with_precision(
                        model, tokenizer, model_name, device, df, target_seq, dtype=torch.bfloat16
                    )
                    bf16_logits = [l.cpu() if l is not None else None for l in bf16_logits]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    fp16_logits_mse, fp16_logits_max_diff = compare_logits(fp32_logits, fp16_logits)
                    bf16_logits_mse, bf16_logits_max_diff = compare_logits(fp32_logits, bf16_logits)
                    
                    # Compare scores
                    fp16_spearman = compare_scores(fp32_scores, fp16_scores)
                    bf16_spearman = compare_scores(fp32_scores, bf16_scores)
                    
                    print(f"\nResults for {dms_id}:")
                    print(f"FP16 Logits - MSE: {fp16_logits_mse:.6e}, Max Abs Diff: {fp16_logits_max_diff:.6e}")
                    print(f"BF16 Logits - MSE: {bf16_logits_mse:.6e}, Max Abs Diff: {bf16_logits_max_diff:.6e}")
                    print(f"FP16 Scores - Spearman: {fp16_spearman:.6f}")
                    print(f"BF16 Scores - Spearman: {bf16_spearman:.6f}")
                    
                    results_summary.append({
                        'model': model_name,
                        'dms_id': dms_id,
                        'seed': seed,
                        'fp16_logits_mse': fp16_logits_mse,
                        'fp16_logits_max_diff': fp16_logits_max_diff,
                        'bf16_logits_mse': bf16_logits_mse,
                        'bf16_logits_max_diff': bf16_logits_max_diff,
                        'fp16_spearman': fp16_spearman,
                        'bf16_spearman': bf16_spearman,
                    })
                    
                    # Collect scores for combined histogram
                    all_fp32_scores.extend(fp32_scores)
                    all_fp16_scores.extend(fp16_scores)
                    all_bf16_scores.extend(bf16_scores)
                    assays_processed += 1
                    
                    # Clean up logits after processing each DMS assay
                    del fp32_logits, fp16_logits, bf16_logits
                    del fp32_scores, fp16_scores, bf16_scores
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error processing {dms_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clean up on error as well
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            hist_path = os.path.join(output_dir, f"{model_name}_error_histogram.png")
            plot_error_histogram(all_fp32_scores, all_fp16_scores, all_bf16_scores, hist_path, model_name, assays_processed)
            print(f"\n  Saved combined histogram to {hist_path}")
            
            # Clean up model
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "precision_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*80}\n")
    
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))


# =============================================================================
# Embedding Precision Testing
# =============================================================================

class PrecisionEmbedder(Embedder):
    """
    Subclass of Embedder that supports precision control via autocast.
    
    Overrides _embed_sequences to wrap forward pass with torch.autocast()
    when dtype is fp16/bf16. For FP32, no autocast is used.
    """
    
    def __init__(self, args: EmbeddingArguments, all_seqs: List[str], dtype: Optional[torch.dtype] = None):
        super().__init__(args, all_seqs)
        self.precision_dtype = dtype
        self.use_autocast = dtype is not None
    
    @torch.inference_mode()
    def _embed_sequences(
            self,
            to_embed: List[str],
            save_path: str,
            embedding_model: Any,
            tokenizer: Any,
            embeddings_dict: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:

        os.makedirs(self.embedding_save_dir, exist_ok=True)
        
        # For FP32, ensure model is in float32
        if self.precision_dtype is None:
            model = embedding_model.float().to(self.device).eval()
        else:
            model = embedding_model.to(self.device).eval()
        
        device = self.device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        collate_fn = build_collator(tokenizer)
        
        if self.matrix_embed:
            pooler = None
        else:
            pooler = Pooler(self.pooling_types)

        def _get_embeddings(
                residue_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                attentions: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
            if residue_embeddings.ndim == 2 or self.matrix_embed:
                return residue_embeddings
            else:
                return pooler(emb=residue_embeddings, attention_mask=attention_mask, attentions=attentions)

        dataset = SimpleProteinDataset(to_embed)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=dataloader_generator(get_global_seed())
        )

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
            seqs = to_embed[i * self.batch_size:(i + 1) * self.batch_size]
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            if 'attention_mask' in batch:
                attention_mask = batch['attention_mask']
            elif 'sequence_ids' in batch:
                attention_mask = (batch['sequence_ids'] != -1).long().to(device)
            else:
                attention_mask = torch.ones_like(batch['input_ids'], device=device)

            # Forward pass with precision control
            if self.use_autocast:
                with torch.autocast(device_type, dtype=self.precision_dtype):
                    residue_embeddings = model(**batch)
            else:
                residue_embeddings = model(**batch)
            
            embeddings = _get_embeddings(residue_embeddings, attention_mask=attention_mask).cpu()

            for seq, emb, mask in zip(seqs, embeddings, attention_mask.cpu()):
                if self.matrix_embed:
                    emb = emb[mask.bool()]
                embeddings_dict[seq] = emb.to(self.embed_dtype)
            
        return embeddings_dict


def load_swissprot_sequences(n_samples: int = 1000, seed: Optional[int] = None, max_length: int = 1022) -> List[str]:
    """
    Load sequences from Synthyra/SwissProt dataset using streaming.
    
    Args:
        n_samples: Number of sequences to sample
        seed: Random seed for reproducibility
        max_length: Maximum sequence length to include
        
    Returns:
        List of sampled sequences
    """
    
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading Synthyra/SwissProt with streaming=True...")
    dataset = load_dataset("Synthyra/SwissProt", split="train", streaming=True)
    
    buffer_size = n_samples * 5
    sequences = []
    
    print(f"Collecting sequences (buffer_size={buffer_size})...")
    for i, example in enumerate(tqdm(dataset, total=buffer_size, desc="Loading sequences")):
        seq = example.get('sequence', example.get('Sequence', ''))
        if seq and len(seq) <= max_length:
            sequences.append(seq)
        if len(sequences) >= buffer_size:
            break
    
    # Sample
    sequences = random.sample(sequences, n_samples)
    
    print(f"Sampled {len(sequences)} sequences (max_length={max_length})")
    return sequences


def compare_embeddings(
    fp32_embs: Dict[str, torch.Tensor],
    other_embs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compare embeddings between FP32 and another precision mode.
    
    Args:
        fp32_embs: Dict of sequence -> FP32 embedding tensor
        other_embs: Dict of sequence -> other precision embedding tensor
        
    Returns:
        Dict with MSE, max absolute difference, and cosine similarity metrics
    """
    all_mse = []
    all_max_diff = []
    all_cos_sim = []
    
    common_seqs = set(fp32_embs.keys()) & set(other_embs.keys())
    
    for seq in common_seqs:
        fp32_emb = fp32_embs[seq].float().flatten()
        other_emb = other_embs[seq].float().flatten()
        
        # Raw difference metrics
        diff = fp32_emb - other_emb
        mse = (diff ** 2).mean().item()
        max_diff = diff.abs().max().item()
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(fp32_emb.unsqueeze(0), other_emb.unsqueeze(0)).item()
        
        all_mse.append(mse)
        all_max_diff.append(max_diff)
        all_cos_sim.append(cos_sim)
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'max_diff': np.max(all_max_diff),
        'cos_sim_mean': np.mean(all_cos_sim),
        'cos_sim_std': np.std(all_cos_sim),
        'cos_sim_min': np.min(all_cos_sim),
        'n_sequences': len(common_seqs),
    }


def plot_embedding_histogram(
    fp32_embs: Dict[str, torch.Tensor],
    fp16_embs: Dict[str, torch.Tensor],
    bf16_embs: Dict[str, torch.Tensor],
    output_path: str,
    model_name: str,
):
    """
    Plot histogram of embedding differences and cosine similarities.
    
    Args:
        fp32_embs: Dict of sequence -> FP32 embedding tensor
        fp16_embs: Dict of sequence -> FP16 embedding tensor
        bf16_embs: Dict of sequence -> BF16 embedding tensor
        output_path: Path to save the plot
        model_name: Model name for title
    """
    common_seqs = set(fp32_embs.keys()) & set(fp16_embs.keys()) & set(bf16_embs.keys())
    
    fp16_diffs = []
    bf16_diffs = []
    fp16_cos_sims = []
    bf16_cos_sims = []
    
    for seq in common_seqs:
        fp32_emb = fp32_embs[seq].float().flatten()
        fp16_emb = fp16_embs[seq].float().flatten()
        bf16_emb = bf16_embs[seq].float().flatten()
        
        # Mean absolute differences per embedding
        fp16_diffs.append((fp32_emb - fp16_emb).abs().mean().item())
        bf16_diffs.append((fp32_emb - bf16_emb).abs().mean().item())
        
        # Cosine similarities
        fp16_cos_sims.append(F.cosine_similarity(fp32_emb.unsqueeze(0), fp16_emb.unsqueeze(0)).item())
        bf16_cos_sims.append(F.cosine_similarity(fp32_emb.unsqueeze(0), bf16_emb.unsqueeze(0)).item())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Absolute Differences
    ax1 = axes[0]
    ax1.hist(fp16_diffs, bins=50, edgecolor='black', alpha=0.6, label='FP16', color='blue')
    ax1.hist(bf16_diffs, bins=50, edgecolor='black', alpha=0.6, label='BF16', color='red')
    ax1.set_xlabel('Mean Absolute Difference per Embedding', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Embedding Differences vs FP32\n{model_name}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add stats
    fp16_mean = np.mean(fp16_diffs)
    bf16_mean = np.mean(bf16_diffs)
    stats_text = (f'FP16 - Mean: {fp16_mean:.6f}, Std: {np.std(fp16_diffs):.6f}\n'
                  f'BF16 - Mean: {bf16_mean:.6f}, Std: {np.std(bf16_diffs):.6f}')
    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # Plot 2: Cosine Similarities
    ax2 = axes[1]
    ax2.hist(fp16_cos_sims, bins=50, edgecolor='black', alpha=0.6, label='FP16', color='blue')
    ax2.hist(bf16_cos_sims, bins=50, edgecolor='black', alpha=0.6, label='BF16', color='red')
    ax2.set_xlabel('Cosine Similarity with FP32', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Embedding Cosine Similarity vs FP32\n{model_name}', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add stats
    fp16_cos_mean = np.mean(fp16_cos_sims)
    bf16_cos_mean = np.mean(bf16_cos_sims)
    stats_text = (f'FP16 - Mean: {fp16_cos_mean:.6f}, Min: {np.min(fp16_cos_sims):.6f}\n'
                  f'BF16 - Mean: {bf16_cos_mean:.6f}, Min: {np.min(bf16_cos_sims):.6f}')
    ax2.text(0.05, 0.95, stats_text,
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_embeddings_precision_test(
    model_names: Optional[List[str]] = None,
    pooling_types: List[str] = ['mean', 'var'],
    n_samples: int = 1000,
    seed: Optional[int] = None,
    output_dir: str = "precision_test_results",
    batch_size: int = 16,
):
    """
    Run embedding precision comparison test across models.
    
    Args:
        model_names: List of model names to test (None = all supported models)
        pooling_types: Pooling methods to use
        n_samples: Number of sequences to sample from SwissProt
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        batch_size: Batch size for embedding
    """
    seed = set_global_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test models
    all_models = list(ProteinGymScorer.MODEL_CONTEXT_LENGTH.keys())
    test_models = [m for m in all_models if not m.lower().startswith("e1")]
    
    if model_names:
        test_models = [m for m in test_models if m in model_names]
    
    print(f"Testing {len(test_models)} models: {test_models}")
    print(f"Pooling types: {pooling_types}")
    
    # Load SwissProt sequences once
    sequences = load_swissprot_sequences(n_samples=n_samples, seed=seed)
    print(f"Loaded {len(sequences)} sequences for embedding")
    
    results_summary = []
    
    for model_name in tqdm(test_models, desc="Models"):
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")
        
        try:
            model, tokenizer = get_base_model(model_name)
            model = model.to(device)
            model.eval()
            
            # Create embedding arguments
            emb_args = EmbeddingArguments(
                embedding_batch_size=batch_size,
                embedding_num_workers=0,
                download_embeddings=False,
                matrix_embed=False,
                embedding_pooling_types=pooling_types,
                save_embeddings=False,
                embed_dtype=torch.float32,
                sql=False,
                embedding_save_dir=output_dir,
            )
            
            # Embed with FP32
            print(f"  Embedding with FP32...")
            set_global_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            fp32_embedder = PrecisionEmbedder(emb_args, sequences, dtype=None)
            fp32_embs = fp32_embedder._embed_sequences(sequences, "", model, tokenizer, {})
            
            # Embed with FP16
            print(f"  Embedding with FP16...")
            set_global_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            fp16_embedder = PrecisionEmbedder(emb_args, sequences, dtype=torch.float16)
            fp16_embs = fp16_embedder._embed_sequences(sequences, "", model, tokenizer, {})
            
            # Embed with BF16
            print(f"  Embedding with BF16...")
            set_global_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            bf16_embedder = PrecisionEmbedder(emb_args, sequences, dtype=torch.bfloat16)
            bf16_embs = bf16_embedder._embed_sequences(sequences, "", model, tokenizer, {})
            
            # Compare embeddings
            fp16_metrics = compare_embeddings(fp32_embs, fp16_embs)
            bf16_metrics = compare_embeddings(fp32_embs, bf16_embs)
            
            print(f"\nResults for {model_name}:")
            print(f"FP16 - MSE: {fp16_metrics['mse_mean']:.6e}, Max Diff: {fp16_metrics['max_diff']:.6e}, "
                  f"Cos Sim: {fp16_metrics['cos_sim_mean']:.6f} (min: {fp16_metrics['cos_sim_min']:.6f})")
            print(f"BF16 - MSE: {bf16_metrics['mse_mean']:.6e}, Max Diff: {bf16_metrics['max_diff']:.6e}, "
                  f"Cos Sim: {bf16_metrics['cos_sim_mean']:.6f} (min: {bf16_metrics['cos_sim_min']:.6f})")
            
            results_summary.append({
                'model': model_name,
                'seed': seed,
                'n_sequences': fp16_metrics['n_sequences'],
                'pooling_types': ','.join(pooling_types),
                'fp16_mse_mean': fp16_metrics['mse_mean'],
                'fp16_max_diff': fp16_metrics['max_diff'],
                'fp16_cos_sim_mean': fp16_metrics['cos_sim_mean'],
                'fp16_cos_sim_min': fp16_metrics['cos_sim_min'],
                'bf16_mse_mean': bf16_metrics['mse_mean'],
                'bf16_max_diff': bf16_metrics['max_diff'],
                'bf16_cos_sim_mean': bf16_metrics['cos_sim_mean'],
                'bf16_cos_sim_min': bf16_metrics['cos_sim_min'],
            })
            
            # Plot histograms
            hist_path = os.path.join(output_dir, f"{model_name}_embedding_precision.png")
            plot_embedding_histogram(fp32_embs, fp16_embs, bf16_embs, hist_path, model_name)
            print(f"  Saved histogram to {hist_path}")
            
            # Clean up
            del fp32_embs, fp16_embs, bf16_embs
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, "embedding_precision_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*80}\n")
    
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(
        description='Precision Comparison Test - Compare FP32, FP16, and BF16 for ProteinGym scoring and embeddings'
    )
    
    # Test mode selection
    parser.add_argument(
        '--embeddings_test',
        action='store_true',
        help='Run embeddings precision test instead of ProteinGym scoring test'
    )
    
    # Common arguments
    parser.add_argument(
        '--model_names',
        nargs='+',
        default=None,
        help='List of model names to test'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='precision_test_results',
        help='Output directory for results'
    )
    
    # ProteinGym scoring test arguments
    parser.add_argument(
        '--dms_ids',
        nargs='+',
        default=TEST_DMS_IDS,
        help='DMS assay IDs to test (for scoring test)'
    )
    
    # Embeddings test arguments
    parser.add_argument(
        '--pooling_type',
        nargs='+',
        default=['mean', 'var'],
        help='Pooling method(s) for embeddings (default: mean var)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of sequences to sample from SwissProt (default: 1000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding (default: 16)'
    )
    
    args = parser.parse_args()
    
    if args.embeddings_test:
        run_embeddings_precision_test(
            model_names=args.model_names,
            pooling_types=args.pooling_type,
            n_samples=args.n_samples,
            seed=args.seed,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        run_precision_test(
            dms_ids=args.dms_ids,
            model_names=args.model_names,
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
