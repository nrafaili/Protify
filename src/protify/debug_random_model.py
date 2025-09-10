#!/usr/bin/env python3
"""
Debug script to identify the Random model determinism issue.
This will help us understand why Random models produce different embeddings
even with the same seed and deterministic settings.
"""
import os
import torch
import sys
sys.path.append('src')

from seed_utils import set_global_seed, set_determinism, is_deterministic, get_global_seed
from base_models.random import build_random_model

# Set CUBLAS workspace config before importing torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def debug_random_model(seed=42, run_name="Run"):
    print(f"\n=== {run_name} ===")
    
    # Set seed and determinism
    actual_seed = set_global_seed(seed)
    #set_determinism()
    
    print(f"Set seed: {actual_seed}")
    print(f"Global seed from get_global_seed(): {get_global_seed()}")
    print(f"is_deterministic(): {is_deterministic()}")
    
    # Create Random model
    print("\nCreating Random model...")
    model, tokenizer = build_random_model('Random')
    print(f"Model deterministic flag: {model.deterministic}")
    
    # Test a simple sequence
    test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    tokenized = tokenizer(test_seq, return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    
    print(f"Input shape: {input_ids.shape}")
    
    # Generate embeddings multiple times to check consistency within same model
    with torch.no_grad():
        emb1 = model(input_ids, attention_mask)
        emb2 = model(input_ids, attention_mask)
        emb3 = model(input_ids, attention_mask)
    
    print(f"Embedding shape: {emb1.shape}")
    print(f"Same model, call 1 vs 2 - Max diff: {torch.max(torch.abs(emb1 - emb2)):.6f}")
    print(f"Same model, call 1 vs 3 - Max diff: {torch.max(torch.abs(emb1 - emb3)):.6f}")
    
    return emb1, model

def main():
    print("Testing Random model determinism...")
    
    # Run 1
    emb1, model1 = debug_random_model(seed=42, run_name="Run 1")
    
    # Run 2 - same seed
    emb2, model2 = debug_random_model(seed=42, run_name="Run 2")
    
    # Compare between runs
    print(f"\n=== Cross-Run Comparison ===")
    max_diff = torch.max(torch.abs(emb1 - emb2))
    mean_diff = torch.mean(torch.abs(emb1 - emb2))
    print(f"Run 1 vs Run 2 - Max diff: {max_diff:.6f}")
    print(f"Run 1 vs Run 2 - Mean diff: {mean_diff:.6f}")
    print(f"Are they close? {torch.allclose(emb1, emb2, atol=1e-6, rtol=1e-5)}")
    
    # Check generator states
    print(f"\n=== Generator State Check ===")
    print(f"Model 1 generator: {model1.generator}")
    print(f"Model 2 generator: {model2.generator}")
    
    # Manual generator test
    print(f"\n=== Manual Generator Test ===")
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)
    
    rand1 = torch.randn(2, 10, 768, generator=gen1)
    rand2 = torch.randn(2, 10, 768, generator=gen2)
    print(f"Manual generator test - Max diff: {torch.max(torch.abs(rand1 - rand2)):.6f}")

if __name__ == "__main__":
    main()
