"""
Precision Comparison Test Suite

Tests masked marginal scoring across FP32, FP16, and BF16 precision modes.
Compares logits and final masked marginal Δlog-prob scores across FP32, FP16, and BF16.
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from benchmarks.proteingym.scorer import ProteinGymScorer, SequenceProcessor
from benchmarks.proteingym.data_loader import load_proteingym_dms
from base_models.get_base_models import get_base_model
from seed_utils import set_global_seed, set_determinism

TEST_DMS_IDS = [
    "A4_HUMAN_Seuma_2022",  # Stability
    "ACE2_HUMAN_Chan_2020",  # Binding
    "D7PM05_CLYGR_Somermeyer_2022",  # Activity
    "ENV_HV1BR_Haddox_2016",  # Organismal fitness
]


def masked_marginal_scoring(
    model: Any,
    tokenizer: Any,
    model_name: str,
    device: torch.device,
    sequences: List[str],
    positions_list: List[List[int]],
    dtype: Optional[torch.dtype] = None,
    max_batch_tokens: int = 16384,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run forward pass with specified precision and return both logits and log_probs.
    
    Args:
        model: The model to run inference on
        tokenizer: The tokenizer
        model_name: Name of the model (for GLM2 handling)
        device: Device to run on
        sequences: List of sequences to score
        positions_list: List of position lists (one per sequence)
        dtype: Optional dtype for autocast (None for FP32)
        max_batch_tokens: Maximum tokens per batch
        
    Returns:
        Tuple of (list of logits tensors, list of log_probs tensors)
    """
    GLM2_MODELS = ["GLM2-150", "GLM2-650", "GLM2-GAIA"]
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for idx, seq in enumerate(sequences):
        if model_name in GLM2_MODELS:
            seq_tokens = len(seq) + 1
        else:
            seq_tokens = len(seq) + 2
        
        if current_batch and current_tokens + seq_tokens > max_batch_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        current_batch.append(idx)
        current_tokens += seq_tokens
    
    if current_batch:
        batches.append(current_batch)
    
    all_logits = [None] * len(sequences)
    all_log_probs = [None] * len(sequences)
    
    dev = "cuda" if device.type == "cuda" else "cpu"
    
    for batch_idx_list in batches:
        batch_sequences = [sequences[i] for i in batch_idx_list]
        batch_positions_list = [positions_list[i] for i in batch_idx_list]
        
        tokens = tokenizer(
            batch_sequences,
            return_tensors='pt',
            add_special_tokens=True,
            padding=False,
        )
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        seq_lengths = attention_mask.sum(dim=1)
        
        # GLM2 does not append EOS
        if model_name in GLM2_MODELS:
            expected_lengths = torch.tensor([len(seq) + 1 for seq in batch_sequences], device=seq_lengths.device)
            if not torch.equal(seq_lengths, expected_lengths):
                raise AssertionError("Tokenized length must equal len(sequence)+1 for GLM2 models in the batch")
        else:
            expected_lengths = torch.tensor([len(seq) + 2 for seq in batch_sequences], device=seq_lengths.device)
            if not torch.equal(seq_lengths, expected_lengths):
                raise AssertionError("Tokenized length must equal len(sequence)+2 for all sequences in the batch")
        
        # Get mask token ID
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            mask_id = tokenizer.convert_tokens_to_ids(getattr(tokenizer, 'mask_token', '<mask>'))
        if mask_id is None:
            raise ValueError("Tokenizer has no mask token.")
        
        masked_input_ids = input_ids.clone()
        for batch_idx, positions in enumerate(batch_positions_list):
            token_indices = [pos + 1 for pos in positions]
            masked_input_ids[batch_idx, token_indices] = mask_id
        
        # Run forward pass with specified precision
        if dtype is None: # FP32
            outputs = model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
        else:
            with torch.autocast(dev, dtype=dtype):
                outputs = model(masked_input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()
        
        for batch_idx, (orig_idx, positions) in enumerate(zip(batch_idx_list, batch_positions_list)):
            token_indices = torch.tensor([pos + 1 for pos in positions], device=device, dtype=torch.long)
            selected_logits = logits[batch_idx, token_indices]
            
            # Check for NaN/non-finite values in logits
            if not torch.isfinite(selected_logits).all():
                nan_count = torch.isnan(selected_logits).sum().item()
                inf_count = torch.isinf(selected_logits).sum().item()
                raise ValueError(
                    f"Non-finite values in logits for sequence {orig_idx}: "
                    f"{nan_count} NaNs, {inf_count} Infs"
                )
            
            log_probs = torch.log_softmax(selected_logits, dim=-1)
            
            # Check for NaN/non-finite values in log_probs
            if not torch.isfinite(log_probs).all():
                nan_count = torch.isnan(log_probs).sum().item()
                inf_count = torch.isinf(log_probs).sum().item()
                raise ValueError(
                    f"Non-finite values in log_probs for sequence {orig_idx}: "
                    f"{nan_count} NaNs, {inf_count} Infs"
                )
            
            all_logits[orig_idx] = selected_logits
            all_log_probs[orig_idx] = log_probs
    
    return all_logits, all_log_probs


def score_with_precision(
    scorer: ProteinGymScorer,
    df: pd.DataFrame,
    target_seq: str,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Returns both scores and logits for comparison.
    
    Args:
        scorer: ProteinGymScorer instance
        df: DataFrame with variants to score
        target_seq: Target sequence
        dtype: Optional dtype for autocast (None for FP32)
        
    Returns:
        Tuple of (scores list, logits list)
    """
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
    
    all_logits, all_log_probs = masked_marginal_scoring(
        scorer.model,
        scorer.tokenizer,
        scorer.model_name,
        scorer.device,
        sequences,
        positions_list,
        dtype=dtype,
        max_batch_tokens=scorer.max_batch_tokens,
    )
    
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
    dms_id: str,
    model_name: str,
):
    """
    Plot histogram of score errors (FP32 - FP16 and FP32 - BF16).
    
    Args:
        fp32_scores: List of FP32 scores
        fp16_scores: List of FP16 scores
        bf16_scores: List of BF16 scores
        output_path: Path to save the plot
        dms_id: DMS assay ID
        model_name: Model name
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
    plt.title(f'Distribution of Score Errors\n{model_name} - {dms_id}', fontsize=14)
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
    
    Args:
        dms_ids: List of DMS assay IDs to test
        model_names: List of model names to test
        seed: Random seed for determinism
        output_dir: Directory to save results
    """
    # Set determinism
    seed = set_global_seed(seed)
    set_determinism()
    
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
        
        try:
            model, tokenizer = get_base_model(model_name, masked_lm=True)
            model = model.to(device)
            model.eval()
            
            scorer = ProteinGymScorer(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=32,
            )
            
            for dms_id in tqdm(dms_ids, desc=f"Assays ({model_name})", leave=False):
                print(f"\n  Processing DMS: {dms_id}")
                
                try:
                    df = load_proteingym_dms(dms_id, mode="benchmark", repo_id="GleghornLab/ProteinGym_DMS")
                    if df is None or len(df) == 0:
                        print(f"    Warning: No data for {dms_id}")
                        continue
                    
                    target_seq = df['target_seq'].iloc[0]
                    seq_len = len(target_seq)
                    
                    # Verify sequence fits in context window
                    context_len = scorer.context_length
                    if seq_len > context_len:
                        print(f"    Warning: Sequence length {seq_len} > context length {context_len} for {model_name}")
                        continue
                    
                    # Run scoring with different precisions
                    print(f"    Running FP32...")
                    set_global_seed(seed)  # Reset seed for each precision run
                    fp32_scores, fp32_logits = score_with_precision(scorer, df, target_seq, dtype=None)
                    
                    print(f"    Running FP16...")
                    set_global_seed(seed)
                    fp16_scores, fp16_logits = score_with_precision(scorer, df, target_seq, dtype=torch.float16)
                    
                    print(f"    Running BF16...")
                    set_global_seed(seed)
                    bf16_scores, bf16_logits = score_with_precision(scorer, df, target_seq, dtype=torch.bfloat16)
                    
                    # Compare logits
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
                    
                    # Generate histogram
                    hist_path = os.path.join(output_dir, f"{model_name}_{dms_id}_error_histogram.png")
                    plot_error_histogram(fp32_scores, fp16_scores, bf16_scores, hist_path, dms_id, model_name)
                    print(f"      Saved histogram to {hist_path}")
                    
                except Exception as e:
                    print(f"Error processing {dms_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Clean up model
            del model, tokenizer, scorer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
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


def main():
    
    parser = argparse.ArgumentParser(
        description='Precision Comparison Test - Compare FP32, FP16, and BF16 scoring'
    )
    parser.add_argument(
        '--dms_ids',
        nargs='+',
        default=TEST_DMS_IDS,
    )
    parser.add_argument(
        '--model_names',
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='precision_test_results',
    )
    
    args = parser.parse_args()
    
    run_precision_test(
        dms_ids=args.dms_ids,
        model_names=args.model_names,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()

