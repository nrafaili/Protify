import torch
from typing import Optional, Tuple


def pad_and_concatenate_dimer(
        A: torch.Tensor,
        B: torch.Tensor,
        a_mask: Optional[torch.Tensor] = None,
        b_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given two sequences A and B with masks, pad (if needed) and concatenate them.
    """
    batch_size, L, d = A.size()
    if a_mask is None:
        a_mask = torch.ones(batch_size, L, device=A.device)
    if b_mask is None:
        b_mask = torch.ones(batch_size, L, device=A.device)
    # Compute the maximum (valid) length in the batch.
    max_len = max(
        int(a_mask[i].sum().item() + b_mask[i].sum().item())
        for i in range(batch_size)
    )
    combined = torch.zeros(batch_size, max_len, d, device=A.device)
    combined_mask = torch.zeros(batch_size, max_len, device=A.device)
    for i in range(batch_size):
        a_len = int(a_mask[i].sum().item())
        b_len = int(b_mask[i].sum().item())
        combined[i, :a_len] = A[i, :a_len]
        combined[i, a_len:a_len+b_len] = B[i, :b_len]
        combined_mask[i, :a_len+b_len] = 1
    return combined, combined_mask