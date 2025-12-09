import re
import os
import sys
import subprocess
import time
import gc
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm.auto import tqdm


class SequenceProcessor:
    """Handles sequence slicing and mutation parsing for ProteinGym."""
    
    @staticmethod
    def get_optimal_window(mutation_position_relative: int, seq_len_wo_special: int, model_window: int) -> list[int]:
        """
        Select an optimal sequence window that fits the maximum model context size.
        If the sequence length is less than the maximum context size, the full sequence is returned.
        """
        half_model_window = model_window // 2
        if seq_len_wo_special <= model_window:
            return [0, seq_len_wo_special]
        elif mutation_position_relative < half_model_window:
            return [0, model_window]
        elif mutation_position_relative >= seq_len_wo_special - half_model_window:
            return [seq_len_wo_special - model_window, seq_len_wo_special]
        else:
            return [max(0, mutation_position_relative - half_model_window), 
                    min(seq_len_wo_special, mutation_position_relative + half_model_window)]

    @staticmethod
    def get_sequence_slices(df, target_seq, model_context_len, start_idx=1, 
                            scoring_window="optimal", indel_mode=False):
        """
        Process a dataframe containing mutant triplets (substitutions) or full mutated sequences (indels).
        Returns a processed DMS in which sequences have been sliced to satisfy the maximum context window.
        
        Modified from https://github.com/OATML-Markslab/Tranception
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to be processed
        target_seq : str
            Full reference sequence (wild type)
        model_context_len : int
            Maximum context size for the model
        start_idx : int
            Integer to move to 0-indexing of positions
        scoring_window : str
            Method to slice sequences: "optimal" or "sliding"
        indel_mode : bool
            Flag for scoring insertions and deletions
        """
        len_target_seq = len(target_seq)
        num_mutants = len(df['mutated_seq'])
        df = df.reset_index(drop=True)
        
        if scoring_window == "optimal":
            if not indel_mode:
                df['mutation_barycenter'] = df['mutant'].apply(
                    lambda x: int(np.array([int(mutation[1:-1]) - start_idx for mutation in x.split(':')]).mean())
                )
                df['scoring_optimal_window'] = df['mutation_barycenter'].apply(
                    lambda x: SequenceProcessor.get_optimal_window(x, len_target_seq, model_context_len)
                )
            else:
                df['mutation_barycenter'] = df['mutated_seq'].apply(lambda x: len(x) // 2)
                df['scoring_optimal_window'] = df['mutated_seq'].apply(lambda x: (0, len(x)))
            
            df['window_start'] = df['scoring_optimal_window'].map(lambda x: x[0])
            df['window_end'] = df['scoring_optimal_window'].map(lambda x: x[1])
            del df['scoring_optimal_window'], df['mutation_barycenter']
            
            df['sliced_mutated_seq'] = [
                seq[start:end] 
                for seq, start, end in zip(df['mutated_seq'], df['window_start'], df['window_end'])
            ]
            
            df_wt = df.copy()
            df_wt['mutated_seq'] = target_seq
            assert len(df_wt) == num_mutants, "Number of wild type sequences should be equal to the number of mutants"
            
            if indel_mode:
                df_wt['window_end'] = df_wt['mutated_seq'].map(lambda x: len(x))
            df_wt['sliced_mutated_seq'] = [
                target_seq[start:end] 
                for start, end in zip(df_wt['window_start'], df_wt['window_end'])
            ]
            df = pd.concat([df, df_wt], axis=0)
            df = df.drop_duplicates()
            keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq', 'window_start', 
                                     'window_end', 'sliced_mutated_seq'] if c in df.columns]
            df = df[keep_cols]
            
        elif scoring_window == "sliding":
            if model_context_len is None:
                model_context_len = len_target_seq
            df_list = []
            start = 0
            while start < len_target_seq:
                end = min(start + model_context_len, len_target_seq)
                df_sliced = df.copy()
                df_sliced['sliced_mutated_seq'] = df_sliced['mutated_seq'].map(lambda x: x[start:end])
                df_sliced['window_start'] = [start] * num_mutants
                df_sliced['window_end'] = df_sliced['mutated_seq'].map(lambda x: min(len(x), end))
                df_sliced_wt = df_sliced.copy()
                df_sliced_wt['mutated_seq'] = [target_seq] * num_mutants
                df_sliced_wt['sliced_mutated_seq'] = df_sliced_wt['mutated_seq'].map(lambda x: x[start:end])
                df_sliced_wt['window_end'] = df_sliced_wt['mutated_seq'].map(lambda x: min(len(x), end))
                df_list.append(df_sliced)
                df_list.append(df_sliced_wt)
                start = end
            df_final = pd.concat(df_list, axis=0)
            df = df_final.drop_duplicates()
            keep_cols = [c for c in ['mutant', 'target_seq', 'mutated_seq', 'window_start', 
                                     'window_end', 'sliced_mutated_seq'] if c in df.columns]
            df = df[keep_cols]
            
        return df.reset_index(drop=True)
    
    @staticmethod
    def parse_mutant_string(mutant: str) -> List[Tuple[str, int, str]]:
        """
        Parse a ProteinGym mutant string where each mutation is separated by ':'.
        Example: "I66N:H67T:S73C" -> [("I", 65, "N"), ("H", 66, "T"), ("S", 72, "C")]
        """
        if mutant is None or (isinstance(mutant, float) and np.isnan(mutant)):
            return []
        parts = str(mutant).split(':')
        parsed: List[Tuple[str, int, str]] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = re.match(r"([A-Za-z*])([0-9]+)([A-Za-z*])", p)
            if not m:
                continue
            wt, pos, mt = m.groups()
            # -1 for 0-based indexing
            parsed.append((wt, int(pos) - 1, mt))
        return parsed
    
    @staticmethod
    def find_mismatches(s1: str | np.ndarray, s2: str) -> list[int]:
        assert isinstance(s1, (str, np.ndarray)), f"s1 must be a string or numpy array, got {type(s1)}"
        assert isinstance(s2, str), f"s2 must be a string, got {type(s2)}"
        assert len(s1) == len(s2), f"s1 and s2 must have the same length, got {len(s1)} and {len(s2)}"
        s1_arr = np.frombuffer(s1.encode(), dtype=np.uint8) if isinstance(s1, str) else s1
        s2_arr = np.frombuffer(s2.encode(), dtype=np.uint8)
        return np.where(s1_arr != s2_arr)[0]

    @staticmethod
    def aa_to_token_ids(tokenizer) -> Dict[str, int]:
        """Precompute amino acid to token ID mapping."""
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        aa_to_id = {}
        for aa in amino_acids:
            token_id = tokenizer.convert_tokens_to_ids(aa)
            if token_id is not None:
                aa_to_id[aa] = token_id
        return aa_to_id


class ProteinGymScorer:
    """Scores protein variants using various scoring methods.
    
    Parameters
    ----------
    model_name : str
    model : Any
    tokenizer : Any
    device : torch.device
    batch_size : int
    use_autocast : bool
        Whether to use autocast for inference (default True)
    dtype : torch.dtype, optional
        Data type for autocast. If None, defaults to float16.
    """
    
    # Model context lengths (minus 2 for special tokens)
    MODEL_CONTEXT_LENGTH = {
        'ESM2-8': 1022, 
        'ESM2-35': 1022,
        'ESM2-150': 1022,
        'ESM2-650': 1022,
        'ESM2-3B': 1022,
        'ESMC-300': 2046,
        'ESMC-600': 2046,
        'ProtBert': 1022,
        'ProtBert-BFD': 1022,
        'GLM2-150': 4095,
        'GLM2-650': 4095,
        'DSM-150': 1022,
        'DSM-650': 1022,
        'DPLM-150': 1022,
        'DPLM-650': 1022,
        'DPLM-3B': 1022,
        'Random-Transformer': 1022,
        'AMPLIFY-120': 2046,
        'AMPLIFY-350': 2046,
        'E1-150': 2046,
        'E1-300': 2046,
        'E1-600': 2046,
    }
    
    
    # Models that don't append EOS token
    GLM2_MODELS = ["GLM2-150", "GLM2-650", "GLM2-GAIA"]
    
    def __init__(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        batch_size: int = 32,
        max_batch_tokens: int = 16384,
        use_autocast: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        # Skip aa_to_id for E1, incompatible with E1Scorer
        if not model_name.lower().startswith("e1"):
            self.aa_to_id = SequenceProcessor.aa_to_token_ids(tokenizer)
        else:
            self.aa_to_id = None
        self.unk_id = getattr(tokenizer, "unk_token_id", None)
        self.context_length = self.MODEL_CONTEXT_LENGTH.get(model_name, 1024)
        self.max_batch_tokens = max_batch_tokens
        
        # Autocast settings
        self.use_autocast = use_autocast
        self.dtype = dtype if dtype is not None else torch.float16

    def score_substitutions(
        self,
        df: pd.DataFrame,
        scoring_method: str = "masked_marginal",
        scoring_window: str = "optimal",
    ) -> pd.DataFrame:
        """Score substitution variants.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with variants to score
        scoring_method : str
            One of: "masked_marginal", "mutant_marginal", "wildtype_marginal", "pll", "global_log_prob"
        scoring_window : str
            "optimal" or "sliding"
        Returns
        -------
        pd.DataFrame
            Output dataframe with 'delta_log_prob' column added
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        # Use E1Scorer for E1 models
        if self.model_name.lower().startswith("e1") and scoring_method == "masked_marginal":
            return self._score_with_e1(df, scoring_method)
        
        # Get sliced sequences
        target_seq = df['target_seq'].iloc[0]
        seq_len = len(target_seq)

        sliced_df = SequenceProcessor.get_sequence_slices(
                df,
                target_seq=target_seq,
                model_context_len=self.context_length,
                start_idx=1,
                scoring_window=scoring_window,
                indel_mode=False
            )
        
        encoded_target = np.frombuffer(target_seq.encode(), dtype=np.uint8)
        mutation_info = {}
        for row in df.itertuples(index=False):
            mutant = row.mutant
            if mutant in mutation_info:  # in case df has duplicates
                continue
            mutated_seq = row.mutated_seq
            mismatches = np.array(
                SequenceProcessor.find_mismatches(encoded_target, mutated_seq),
                dtype=np.int64
            )
            wt_aas = ''.join(target_seq[p] for p in mismatches)
            mt_aas = ''.join(mutated_seq[p] for p in mismatches)
            mutation_info[mutant] = (mismatches, wt_aas, mt_aas)
    
        # --- window_info (FAST) ---
        if scoring_window == "optimal" and seq_len <= self.context_length:
            # no slicing needed; everyone shares one window
            uniq_mutants = pd.unique(df["mutant"])
            window_info = {
                m: {"window_start": 0, "window_end": seq_len, "sliced_seq": target_seq}
                for m in uniq_mutants
            }
        else:
            wt_rows = (
                sliced_df.loc[sliced_df["mutated_seq"].eq(target_seq),
                                ["mutant", "window_start", "window_end", "sliced_mutated_seq"]]
                .drop_duplicates(subset=["mutant"])
            )
            window_info = {
                r.mutant: {
                    "window_start": int(r.window_start),
                    "window_end": int(r.window_end),
                    "sliced_seq": r.sliced_mutated_seq,
                }
                for r in wt_rows.itertuples(index=False)
            }
        
        if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
            scores = self._score_marginal(df, target_seq, sliced_df, scoring_method, mutation_info, window_info)
        elif scoring_method == "pll":
            scores = self._score_pll_substitutions(sliced_df, target_seq)
        else:  # global_log_prob
            scores = self._score_global_log_prob(sliced_df, target_seq)
        
        out = df.copy()
        if scoring_method in ["masked_marginal", "mutant_marginal", "wildtype_marginal"]:
            out['delta_log_prob'] = scores
        else:
            out['delta_log_prob'] = out['mutant'].map(scores)
        return out
    
    def score_indels(
        self,
        df: pd.DataFrame,
        scoring_window: str = "sliding",
    ) -> pd.DataFrame:
        """Score indel variants using PLL.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with indel variants to score
        scoring_window : str
            Sliding window mode should be used for indels scoring
        Returns
        -------
        pd.DataFrame
            Output dataframe with 'delta_log_prob' column added
        """
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame is empty")
        
        target_seq = df['target_seq'].iloc[0]
        model_context_len = self.context_length
        if model_context_len is None:
            model_context_len = len(target_seq)
        
        sliced_df = SequenceProcessor.get_sequence_slices(
            df,
            target_seq=target_seq,
            model_context_len=model_context_len,
            start_idx=1,
            scoring_window='sliding',
            indel_mode=True
        )
        
        seqs_to_score = sliced_df['sliced_mutated_seq'].to_list()
        
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences (indels)...")
        
        iterator = tqdm(
            total=len(seqs_to_score),
            desc="PLL computation (indels)",
            unit="seq",
            position=1,
            leave=False,
        )
        
        pll_results = self._calculate_pll_batched(seqs_to_score, iterator)
        iterator.close()
        
        # Grab normalized PLL for indels
        pll_cache = {seq: result for seq, result in zip(seqs_to_score, pll_results)}
        
        # Add a mapped column of per-window scores, then average by mutated_seq
        sliced_df['window_score'] = sliced_df['sliced_mutated_seq'].map(pll_cache)
        scores_by_variant = (
            sliced_df.groupby('mutated_seq')['window_score']
            .mean()
            .to_dict()
        )
        
        out = df.copy()
        out['delta_log_prob'] = out['mutated_seq'].map(scores_by_variant)
        return out
    

    def _create_dynamic_batches(
        self,
        sequences: List[str],
        max_batch_tokens: Optional[int] = None,
    ) -> List[List[int]]:
        """Create dynamic batches that pack sequences greedily until max_batch_tokens is reached."""
        if max_batch_tokens is None:
            max_batch_tokens = self.max_batch_tokens
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, seq in enumerate(sequences):
            # Calculate tokens for this sequence (+2 for BOS/EOS, or +1 for GLM2)
            if self.model_name in self.GLM2_MODELS:
                seq_tokens = len(seq) + 1
            else:
                seq_tokens = len(seq) + 2
            
            # If adding this sequence would exceed max_batch_tokens, start new batch
            if current_batch and current_tokens + seq_tokens > max_batch_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += seq_tokens
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        return batches
        
    def _score_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        sliced_df: pd.DataFrame,
        scoring_method: str,
        mutation_info: Dict,
        window_info: Dict,
    ) -> List[float]:
        """Score using marginal methods (masked/wildtype/mutant)."""
        
        if scoring_method == "masked_marginal":
            return self._score_masked_marginal(df, target_seq, sliced_df, mutation_info, window_info)
        elif scoring_method == "wildtype_marginal":
            return self._score_wildtype_marginal(df, target_seq, sliced_df, mutation_info, window_info)
        else:
            return self._score_mutant_marginal(df, target_seq, sliced_df, mutation_info, window_info)
    
    def _score_masked_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        sliced_df: pd.DataFrame,
        mutation_info: Dict,
        window_info: Dict,
    ) -> List[float]:
        """Score using masked marginal method with optimized vectorization."""
        # Group by (window_start, window_end, pos_tuple) -> List[(row_idx, positions, wt_aas, mt_aas)]
        position_groups: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, np.ndarray, str, str]]] = {}
        
        for row_idx, row in enumerate(df.itertuples(index=False)):
            mutant = row.mutant
            positions, wt_aas, mt_aas = mutation_info[mutant]
            
            window = window_info.get(mutant)
            if window is None:
                raise ValueError(f"No available window for mutant {mutant}")
            
            window_start = window['window_start']
            window_end = window['window_end']
            
            # Check all positions are within window
            min_pos = positions.min()
            max_pos = positions.max()
            if not (window_start <= min_pos and max_pos < window_end):
                raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
            
            # Convert to relative positions
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
        
        total_variants = len(df)
        print(f"Computing scores for {len(sequences)} inputs, covering {total_variants} variants ...")
        
        per_variant_log_probs = self._position_log_probs_batched(
            "masked_marginal", sequences, positions_list
        )
        
        # Vectorized scoring
        scores = [0.0] * len(df)
        for variants_in_group, log_probs in zip(variant_info, per_variant_log_probs):
            # Pre-collect all WT/MT ids for the group
            num_variants = len(variants_in_group)
            num_positions = log_probs.size(0)
            
            # Build tensors for all variants in this group
            wt_ids_list = []
            mt_ids_list = []
            
            for row_idx, wt_aas, mt_aas in variants_in_group:
                assert len(wt_aas) == num_positions, f"Variant {row_idx} in group has {len(wt_aas)} muts, expected {num_positions}"
                wt_ids = [self.aa_to_id[aa] for aa in wt_aas]
                mt_ids = [self.aa_to_id[aa] for aa in mt_aas]
                wt_ids_list.append(wt_ids)
                mt_ids_list.append(mt_ids)
            
            # Convert to tensors [num_variants, num_positions]
            wt_tensor = torch.tensor(wt_ids_list, device=log_probs.device, dtype=torch.long)
            mt_tensor = torch.tensor(mt_ids_list, device=log_probs.device, dtype=torch.long)
            
            pos_idx = torch.arange(num_positions, device=log_probs.device)[None, :].expand(num_variants, -1)
            wt_log_probs = log_probs[pos_idx, wt_tensor]  # [num_variants, num_positions]
            mt_log_probs = log_probs[pos_idx, mt_tensor]
            deltas = (mt_log_probs - wt_log_probs).sum(dim=1)
            
            # Assign to scores
            for i, (row_idx, _, _) in enumerate(variants_in_group):
                scores[row_idx] = deltas[i].item()
        
        return scores
    
    def _score_wildtype_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        sliced_df: pd.DataFrame,
        mutation_info: Dict,
        window_info: Dict,
    ) -> List[float]:
        """Score using wildtype marginal method."""
        # Group by (window_start, window_end) -> List[(row_idx, positions, wt_aas, mt_aas)]
        window_groups: Dict[Tuple[int, int], List[Tuple[int, np.ndarray, str, str]]] = {}
        
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
            
            key = (window_start, window_end)
            window_groups.setdefault(key, []).append((row_idx, positions, wt_aas, mt_aas))
        
        sequences: List[str] = []
        positions_list: List[List[int]] = []
        window_to_variants: List[List[Tuple[int, np.ndarray, str, str]]] = []
        
        for (window_start, window_end), variants in window_groups.items():
            window_seq = target_seq[window_start:window_end]
            sequences.append(window_seq)
            
            all_positions = set()
            for _, positions, _, _ in variants:
                all_positions.update((positions - window_start).tolist())
            positions_list.append(sorted(all_positions))
            window_to_variants.append(variants)
        
        total_variants = len(df)
        print(f"Computing scores for {len(sequences)} windows, covering {total_variants} variants ...")
        
        per_variant_log_probs = self._position_log_probs_batched(
            "wildtype_marginal", sequences, positions_list
        )
        
        scores = [0.0] * len(df)
        for window_idx, (window_log_probs, variants) in enumerate(zip(per_variant_log_probs, window_to_variants)):
            window_positions = positions_list[window_idx]
            window_start = list(window_groups.keys())[window_idx][0]
            pos_to_idx = {pos: idx for idx, pos in enumerate(window_positions)}
            
            # Vectorized scoring for this window
            wt_ids_list = []
            mt_ids_list = []
            pos_indices_list = []
            row_indices = []
            
            for row_idx, positions, wt_aas, mt_aas in variants:
                pos_rels = positions - window_start
                pos_indices = [pos_to_idx[p] for p in pos_rels]
                
                wt_ids = [self.aa_to_id[aa] for aa in wt_aas]
                mt_ids = [self.aa_to_id[aa] for aa in mt_aas]
                
                wt_ids_list.append(wt_ids)
                mt_ids_list.append(mt_ids)
                pos_indices_list.append(pos_indices)
                row_indices.append(row_idx)
            
            # Score each variant in this window
            for i, (row_idx, pos_indices, wt_ids, mt_ids) in enumerate(zip(row_indices, pos_indices_list, wt_ids_list, mt_ids_list)):
                pos_tensor = torch.tensor(pos_indices, device=window_log_probs.device, dtype=torch.long)
                variant_log_probs = window_log_probs[pos_tensor]  # [num_positions, vocab]
                
                wt_tensor = torch.tensor(wt_ids, device=variant_log_probs.device, dtype=torch.long)
                mt_tensor = torch.tensor(mt_ids, device=variant_log_probs.device, dtype=torch.long)
                
                indices = torch.arange(len(wt_ids), device=variant_log_probs.device)
                deltas = variant_log_probs[indices, mt_tensor] - variant_log_probs[indices, wt_tensor]
                scores[row_idx] = deltas.sum().item()
        
        return scores
    
    def _score_mutant_marginal(
        self,
        df: pd.DataFrame,
        target_seq: str,
        sliced_df: pd.DataFrame,
        mutation_info: Dict,
        window_info: Dict,
    ) -> List[float]:
        """Score using mutant marginal method."""
        sequences: List[str] = []
        positions_list: List[List[int]] = []
        variant_info: List[Tuple[int, str, str]] = []
        
        for row_idx, row in enumerate(df.itertuples(index=False)):
            mutant = row.mutant
            mutated_seq = row.mutated_seq
            positions, wt_aas, mt_aas = mutation_info[mutant]
            
            # Get mutant slice
            mutant_slices = sliced_df[sliced_df['mutant'] == mutant]
            mut_slice = mutant_slices[mutant_slices['mutated_seq'] == mutated_seq]
            if len(mut_slice) == 0:
                raise ValueError(f"No available slice for mutant {mutant}")
            slice_row = mut_slice.iloc[0]
            
            window_start = int(slice_row['window_start'])
            window_end = int(slice_row['window_end'])
            window_seq = slice_row['sliced_mutated_seq']
            
            min_pos = positions.min()
            max_pos = positions.max()
            if not (window_start <= min_pos and max_pos < window_end):
                raise ValueError(f"Window {window_start}-{window_end} does not contain all positions for variant {mutant}")
            
            pos_rels = (positions - window_start).tolist()
            
            sequences.append(window_seq)
            positions_list.append(pos_rels)
            variant_info.append((row_idx, wt_aas, mt_aas))
        
        print(f"Computing scores for {len(sequences)} variants ...")
        
        per_variant_log_probs = self._position_log_probs_batched(
            "mutant_marginal", sequences, positions_list
        )
        
        scores = [0.0] * len(df)
        for (row_idx, wt_aas, mt_aas), log_probs in zip(variant_info, per_variant_log_probs):
            wt_ids = [self.aa_to_id[aa] for aa in wt_aas]
            mt_ids = [self.aa_to_id[aa] for aa in mt_aas]
            
            wt_tensor = torch.tensor(wt_ids, device=log_probs.device, dtype=torch.long)
            mt_tensor = torch.tensor(mt_ids, device=log_probs.device, dtype=torch.long)
            
            indices = torch.arange(len(wt_ids), device=log_probs.device)
            deltas = log_probs[indices, mt_tensor] - log_probs[indices, wt_tensor]
            scores[row_idx] = deltas.sum().item()
        
        return scores
    
    def _score_pll_substitutions(
        self,
        sliced_df: pd.DataFrame,
        target_seq: str,
    ) -> Dict[str, float]:
        """Score substitutions using pseudo-log-likelihood."""
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        
        seqs_to_score = mutated_slices['sliced_mutated_seq'].drop_duplicates().tolist()
        
        print(f"Computing PLL for {len(seqs_to_score)} unique sequences...")
        
        pll_progress = tqdm(
            total=len(seqs_to_score),
            desc="PLL computation",
            unit="seq",
            position=1,
            leave=False,
        )
        
        pll_results = self._calculate_pll_batched(seqs_to_score, pll_progress)
        pll_progress.close()
        
        seq_to_pll = {seq: res[0] for seq, res in zip(seqs_to_score, pll_results)}
        mutated_slices['sequence_pll'] = mutated_slices['sliced_mutated_seq'].map(seq_to_pll)
        
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_pll']
            .first()
            .to_dict()
        )
        
        return scores_by_variant
    
    def _score_global_log_prob(
        self,
        sliced_df: pd.DataFrame,
        target_seq: str,
    ) -> Dict[str, float]:
        """Score using global log probability."""
        mutated_slices = sliced_df[sliced_df['mutated_seq'] != target_seq].copy()
        seqs_to_score = mutated_slices['sliced_mutated_seq'].tolist()
        
        print(f"Computing global log prob for {len(seqs_to_score)} unique sequences...")
        
        iterator = tqdm(
            range(0, len(seqs_to_score), self.batch_size),
            total=(len(seqs_to_score) + self.batch_size - 1) // self.batch_size,
            desc="Global log prob batches",
            unit="batch",
            position=1,
            leave=False,
        )
        
        log_prob_results = self._get_sequence_log_probability_batched(seqs_to_score, iterator)
        
        mutated_slices['sequence_log_prob'] = log_prob_results
        
        scores_by_variant = (
            mutated_slices.groupby('mutant')['sequence_log_prob']
            .first()
            .to_dict()
        )
        
        return scores_by_variant
    
    def _score_with_e1(self, df: pd.DataFrame, scoring_method: str) -> pd.DataFrame:
        """Score variants using E1Scorer."""
        from .e1_scorer import E1Scorer, EncoderScoreMethod
        
        scorer = E1Scorer(model=self.model, method=EncoderScoreMethod.MASKED_MARGINAL)
        # E1 has a context length of 8192, so we don't need to slice the sequences for scoring with these models
        target_seq = df['target_seq'].iloc[0]
        sequences = df['mutated_seq'].tolist()
        sequence_ids = df['mutant'].tolist()
        
        print(f"Scoring {len(sequences)} variants with E1 ({scoring_method})...")
        
        try:
            results = scorer.score(
                parent_sequence=target_seq,
                sequences=sequences,
                sequence_ids=sequence_ids,
                context_seqs=None,
                context_reduction="none",
            )
            
            scores_dict = {r["id"]: r["score"] for r in results}
            scores = [scores_dict[mutant] for mutant in sequence_ids]
            
            out = df.copy()
            out['delta_log_prob'] = scores
            return out
        finally:
            # Clean up E1Scorer's cache and GPU memory after each assay
            scorer.cleanup()
            del scorer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def _position_log_probs_batched(
        self,
        scoring_method: str,
        sequences: List[str],
        positions_list: List[List[int]],
        return_logits: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Return batched log probabilities with dynamic batching."""
        assert len(sequences) == len(positions_list), "Must have one position list per sequence"
        
        # Create dynamic batches
        batch_indices = self._create_dynamic_batches(sequences)
        
        all_log_probs = [None] * len(sequences)
        all_selected_logits = [None] * len(sequences) if return_logits else None
        
        progress_bar = tqdm(
            batch_indices,
            desc=f"Assay batches ({scoring_method})",
            unit="batch",
            position=1,
            leave=False,
        )
        
        dev = "cuda" if self.device.type == "cuda" else "cpu"
        
        for batch_idx_list in progress_bar:
            batch_sequences = [sequences[i] for i in batch_idx_list]
            batch_positions_list = [positions_list[i] for i in batch_idx_list]
            
            tokens = self.tokenizer(
                batch_sequences,
                return_tensors='pt',
                add_special_tokens=True,
                padding='longest',
            )
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            seq_lengths = attention_mask.sum(dim=1)
            
            # GLM2 does not append EOS
            if self.model_name in self.GLM2_MODELS:
                expected_lengths = torch.tensor([len(seq) + 1 for seq in batch_sequences], device=seq_lengths.device)
                if not torch.equal(seq_lengths, expected_lengths):
                    raise AssertionError("Tokenized length must equal len(sequence)+1 for GLM2 models in the batch")
            else:
                expected_lengths = torch.tensor([len(seq) + 2 for seq in batch_sequences], device=seq_lengths.device)
                if not torch.equal(seq_lengths, expected_lengths):
                    raise AssertionError("Tokenized length must equal len(sequence)+2 for all sequences in the batch")
            
            if scoring_method == "masked_marginal":
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None:
                    mask_id = self.tokenizer.convert_tokens_to_ids(getattr(self.tokenizer, 'mask_token', '<mask>'))
                if mask_id is None:
                    raise ValueError("Tokenizer has no mask token.")
                
                masked_input_ids = input_ids.clone()
                for batch_idx, positions in enumerate(batch_positions_list):
                    token_indices = [pos + 1 for pos in positions]
                    masked_input_ids[batch_idx, token_indices] = mask_id
                
                if self.use_autocast:
                    with torch.autocast(dev, dtype=self.dtype):
                        outputs = self.model(masked_input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(masked_input_ids, attention_mask=attention_mask)
            else:
                if self.use_autocast:
                    with torch.autocast(dev, dtype=self.dtype):
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits.float()
            
            for batch_idx, (orig_idx, positions) in enumerate(zip(batch_idx_list, batch_positions_list)):
                token_indices = torch.tensor([pos + 1 for pos in positions], device=self.device, dtype=torch.long)
                selected_logits = logits[batch_idx, token_indices]
                log_probs = torch.log_softmax(selected_logits, dim=-1)
                all_log_probs[orig_idx] = log_probs
                if return_logits:
                    all_selected_logits[orig_idx] = selected_logits
        
        if return_logits:
            return all_log_probs, all_selected_logits  # type: ignore[return-value]
        return all_log_probs
    
    @torch.inference_mode()
    def _calculate_pll_batched(
        self,
        sequences: List[str],
        progress_bar,
    ) -> List[Tuple[float, float]]:
        """Calculate pseudo-log-likelihood for multiple sequences with batched processing."""
        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            mask_id = self.tokenizer.convert_tokens_to_ids(getattr(self.tokenizer, 'mask_token', '<mask>'))
        if mask_id is None:
            raise ValueError("Tokenizer must provide a valid mask token id")
        
        # Group sequences by length for efficient batching
        length_groups = defaultdict(list)
        for idx, seq in enumerate(sequences):
            length_groups[len(seq)].append((idx, seq))
        
        results = [None] * len(sequences)
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        
        for seq_len, indexed_seqs in length_groups.items():
            indices = [idx for idx, _ in indexed_seqs]
            seqs = [seq for _, seq in indexed_seqs]
            
            tokens = self.tokenizer(seqs, return_tensors="pt", add_special_tokens=True, padding=False)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            num_seqs = input_ids.size(0)
            
            if self.model_name in self.GLM2_MODELS:
                expected_len = seq_len + 1
                assert input_ids.shape[1] == expected_len, (
                    f"Tokenized length {input_ids.shape[1]} must equal len(sequence)+1 ({expected_len}) for GLM2"
                )
            else:
                expected_len = seq_len + 2
                assert input_ids.shape[1] == expected_len, (
                    f"Tokenized length {input_ids.shape[1]} must equal len(sequence)+2 ({expected_len})"
                )
            
            seq_start = 1
            if self.model_name in self.GLM2_MODELS:
                seq_end = input_ids.size(1)
            else:
                seq_end = input_ids.size(1) - 1
            positions = list(range(seq_start, seq_end))
            L = len(positions)
            
            total_lls = torch.zeros(num_seqs, device=self.device)
            
            for batch_start_idx in range(0, len(positions), self.batch_size):
                batch_end_idx = min(batch_start_idx + self.batch_size, len(positions))
                batch_positions = positions[batch_start_idx:batch_end_idx]
                num_positions = len(batch_positions)
                
                masked_batch = input_ids.unsqueeze(1).expand(-1, num_positions, -1).reshape(num_seqs * num_positions, -1).clone()
                attention_mask_batch = attention_mask.unsqueeze(1).expand(-1, num_positions, -1).reshape(num_seqs * num_positions, -1)
                
                position_tensor = torch.tensor(batch_positions, device=self.device)
                row_indices = torch.arange(num_seqs * num_positions, device=self.device)
                pos_indices = position_tensor.repeat(num_seqs)
                masked_batch[row_indices, pos_indices] = mask_id
                
                if self.use_autocast:
                    with torch.autocast(device_type, dtype=self.dtype):
                        outputs = self.model(masked_batch, attention_mask=attention_mask_batch)
                else:
                    outputs = self.model(masked_batch, attention_mask=attention_mask_batch)
                logits = outputs.logits.float()
                
                log_probs = torch.log_softmax(logits, dim=-1)
                
                true_ids = input_ids[:, batch_positions]
                true_ids_flat = true_ids.reshape(-1)
                
                batch_indices = torch.arange(num_seqs * num_positions, device=self.device)
                selected_log_probs = log_probs[batch_indices, pos_indices, true_ids_flat]
                
                selected_log_probs = selected_log_probs.reshape(num_seqs, num_positions)
                total_lls += selected_log_probs.sum(dim=1)
            
            progress_bar.update(num_seqs)
            
            for i, orig_idx in enumerate(indices):
                total_ll = total_lls[i].item()
                results[orig_idx] = (total_ll, total_ll / L)
        
        return results
    
    @torch.inference_mode()
    def _get_sequence_log_probability_batched(
        self,
        sequences: List[str],
        progress_bar,
    ) -> List[float]:
        """Compute log probability for multiple sequences with batched processing."""
        results = []
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + self.batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            
            tokens = self.tokenizer(
                batch_sequences,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,
            )
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            if self.use_autocast:
                with torch.autocast(device_type, dtype=self.dtype):
                    output = self.model(input_ids, attention_mask=attention_mask)
            else:
                output = self.model(input_ids, attention_mask=attention_mask)
            logits = output.logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            
            selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
            masked_log_probs = selected_log_probs * attention_mask
            seq_log_probs = masked_log_probs.sum(dim=1)
            
            results.extend(seq_log_probs.tolist())
        
        return results


class ProteinGymRunner:
    """Orchestrates ProteinGym zero-shot scoring across models and assays.
    
    Parameters
    ----------
    results_dir : str
        Directory to save results
    repo_id : str
        HuggingFace repo ID for ProteinGym data
    device : str, optional
        Device to run on (defaults to CUDA if available)
    """
    
    def __init__(
        self,
        results_dir: str,
        repo_id: str = "GleghornLab/ProteinGym_DMS",
        device: Optional[str] = None,
    ):
        self.results_dir = results_dir
        self.repo_id = repo_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        os.makedirs(results_dir, exist_ok=True)
    
    def run(
        self,
        dms_ids: List[str],
        model_names: List[str],
        mode: str = "benchmark",
        scoring_method: str = "masked_marginal",
        scoring_window: str = "optimal",
        batch_size: int = 32,
        max_batch_tokens: int = 65536,
        use_autocast: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, float]:
        """Run zero-shot scoring for all specified models and assays.
        
        Parameters
        ----------
        dms_ids : List[str]
            List of DMS assay IDs to score
        model_names : List[str]
            List of model names to use for scoring
        mode : str
            One of: "benchmark", "indels", "singles", "multiples"
        scoring_method : str
            Scoring method to use
        scoring_window : str
            "optimal" or "sliding"
        batch_size : int
            Batch size for inference
        max_batch_tokens : int
            Maximum tokens per batch for dynamic batching (default 65536)
        use_autocast : bool
            Whether to use autocast for inference (default True)
        dtype : torch.dtype, optional
            Data type for autocast. If None, uses model default.
            
        Returns
        -------
        Dict[str, float]
            Mapping of model_name -> elapsed_time
        """
        from base_models.get_base_models import get_base_model
        from .data_loader import load_proteingym_dms
        
        timing = {}
        
        for model_name in model_names:
            start_time = time.time()
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            # Clear torch compilation cache for E1 models
            if model_name.lower().startswith('e1'):
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
            
            model, tokenizer = get_base_model(model_name, masked_lm=True)
            model = model.to(self.device)
            
            scorer = ProteinGymScorer(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                use_autocast=use_autocast,
                dtype=dtype,
            )
            
            assay_iterator = tqdm(dms_ids, desc="All assays", unit="assay", position=0)
            
            for dms_id in assay_iterator:
                df = load_proteingym_dms(dms_id, mode=mode, repo_id=self.repo_id)
                if df is None or len(df) == 0:
                    raise ValueError(f"No data found for DMS ID: {dms_id}")
                
                assay_iterator.set_description_str(f"Assay {dms_id}")
                
                if mode == 'indels':
                    results_df = scorer.score_indels(
                        df,
                        scoring_window='sliding',
                    )
                    suffix = 'pll'
                else:
                    results_df = scorer.score_substitutions(
                        df,
                        scoring_method=scoring_method,
                        scoring_window=scoring_window,
                    )
                    suffix = scoring_method
                
                self._save_results(dms_id, results_df, model_name, suffix, mode)
                tqdm.write(f"[Assay {dms_id}] saved/updated")
            
            del scorer
            del model
            del tokenizer
            
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Run garbage collection
            gc.collect()
            
            # Clear torch compilation cache for E1 models to prevent shape mismatches
            if model_name.lower().startswith('e1'):
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
            
            tqdm.write(f"Model {model_name} deleted and memory cleared")
            
            timing[model_name] = time.time() - start_time
        
        return timing
    
    def _save_results(
        self,
        dms_id: str,
        results_df: pd.DataFrame,
        model_name: str,
        suffix: str,
        mode: str,
    ):
        """Save/merge results for a DMS assay."""
        per_dms_path = os.path.join(self.results_dir, f"{dms_id}_zs_{suffix}.csv")
        
        results_to_save = results_df.copy()
        if 'delta_log_prob' in results_to_save.columns:
            results_to_save = results_to_save.rename(columns={'delta_log_prob': model_name})
        
        if 'target_seq' in results_to_save.columns:
            results_to_save = results_to_save.drop(columns=['target_seq'])
        
        # If an aggregated file exists, merge the new model column
        if os.path.exists(per_dms_path):
            try:
                existing = pd.read_csv(per_dms_path)
                if 'mutant' in existing.columns:
                    merged = existing.merge(
                        results_to_save[['mutant', model_name]],
                        on='mutant',
                        how='outer',
                    )
                elif 'mutated_seq' in existing.columns:  # for indels
                    merged = existing.merge(
                        results_to_save[['mutated_seq', model_name]],
                        on='mutated_seq',
                        how='outer',
                    )
                merged.to_csv(per_dms_path, index=False)
            except Exception as e:
                print(f"Error merging results for {dms_id}: {e}")
                results_to_save.to_csv(per_dms_path, index=False)
        else:
            results_to_save.to_csv(per_dms_path, index=False)
    
    def run_benchmark(
        self,
        model_names: List[str],
        dms_ids: List[str],
        mode: str,
        scoring_method: str,
    ):
        """Run the ProteinGym benchmarking script on scored CSV files.
        
        Parameters
        ----------
        model_names : List[str]
            List of model names to evaluate
        dms_ids : List[str]
            List of DMS assay IDs to evaluate
        mode : str
            Mode: 'benchmark', 'indels', 'singles', 'multiples'
        scoring_method : str
            Scoring method used (e.g., 'masked_marginal', 'pll')
        """
        try:
            pg_dir = os.path.join(os.path.dirname(__file__))
            reference_mapping = os.path.join(pg_dir, 'DMS_substitutions.csv')
            config_path = os.path.join(pg_dir, 'config.json')
            perf_out_dir = os.path.join(self.results_dir, 'benchmark_performance')
            os.makedirs(perf_out_dir, exist_ok=True)

            script_path = os.path.join(pg_dir, 'DMS_benchmark_performance.py')
            script_cmd = [
                sys.executable, script_path,
                '--input_scoring_files_folder', self.results_dir,
                '--output_performance_file_folder', perf_out_dir,
                '--DMS_reference_file_path', reference_mapping,
                '--config_file', config_path,
                '--performance_by_depth',
            ]
            script_cmd += ['--scoring_method', scoring_method]
            if isinstance(model_names, (list, tuple)) and len(model_names) > 0:
                script_cmd += ['--selected_model_names', *model_names]
            if isinstance(dms_ids, (list, tuple)) and len(dms_ids) > 0:
                script_cmd += ['--dms_ids', *[str(x) for x in dms_ids]]
            if isinstance(mode, str) and mode.lower() == 'indels':
                script_cmd.append('--indel_mode')
            subprocess.run(script_cmd, check=True)
            
            print(f"Benchmark performance computed. Outputs in {perf_out_dir}")
        except Exception as e:
            print(f"Failed to compute benchmark performance: {e}")
    
    @staticmethod
    def collect_spearman(results_dir: str, model_names: List[str]) -> Dict[str, float]:
        """Parse ProteinGym benchmark Summary CSV and return {model_name: spearman}.
        
        Looks for Summary_performance_DMS_[substitutions|indels]_Spearman.csv and
        creates a dictionary of {model_name: spearman} for the given model names.
        This is used to pass Spearman scores to the visualization module for plotting.
        """
        perf_out_dir = os.path.join(results_dir, 'benchmark_performance')
        spearman_dir = os.path.join(perf_out_dir, 'Spearman')
        sub_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_substitutions_Spearman.csv')
        ind_csv = os.path.join(spearman_dir, 'Summary_performance_DMS_indels_Spearman.csv')
        csv_path = sub_csv if os.path.exists(sub_csv) else ind_csv if os.path.exists(ind_csv) else None
        
        if csv_path is None:
            print(f"ProteinGym Spearman summary not found in {spearman_dir}")
            return {}
        
        df = pd.read_csv(csv_path)
        if 'Model_name' not in df.columns or 'Average_Spearman' not in df.columns:
            print("ProteinGym summary CSV missing required columns: 'Model_name' and 'Average_Spearman'")
            return {}
        
        model_scores = {}
        for _, row in df.iterrows():
            try:
                name = str(row['Model_name'])
                score = float(row['Average_Spearman'])
            except Exception:
                continue
            model_scores[name] = score
        
        out = {}
        for model_name in (model_names or []):
            if model_name in model_scores:
                out[model_name] = float(model_scores[model_name])
        return out