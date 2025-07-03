"""
Example script demonstrating how to use the inference pipeline for hydrolase screening.

This script shows two main workflows:
1. Training a hydrolase classifier
2. Using the trained model to screen TrEMBL for hydrolases
"""

import os
from inference import HydrolaseInference, InferenceArguments, run_trembl_screening
from main import MainProcess
import argparse
from types import SimpleNamespace


def train_hydrolase_model():
    """Example of training a hydrolase classification model."""
    
    # Set up training arguments
    args = SimpleNamespace(
        # Data settings
        data_dirs=['/path/to/your/hydrolase/dataset'],
        col_names=['seqs', 'labels'],
        max_length=1024,
        trim=False,
        
        # Model settings
        model_names=['ESM2-650'],
        
        # Probe settings
        probe_type='linear',
        probe_pooling_types=['mean'],
        hidden_dim=8192,
        dropout=0.2,
        
        # Training settings
        num_epochs=100,
        probe_batch_size=32,
        lr=1e-4,
        patience=10,
        save_model=True,  # Important: enable model saving
        model_save_dir='hydrolase_models',
        
        # Embedding settings
        embedding_batch_size=8,
        save_embeddings=True,
        embedding_save_dir='embeddings',
        
        # Paths
        log_dir='logs',
        results_dir='results',
        plots_dir='plots'
    )
    
    # Run training
    process = MainProcess(args)
    process.apply_current_settings()
    process.get_datasets()
    process.save_embeddings_to_disk()  # Pre-compute embeddings
    process.run_nn_probes()  # Train the model
    
    print("Training completed! Model saved in 'hydrolase_models' directory.")


def screen_trembl_simple_example():
    """Simple example of screening TrEMBL sequences."""
    
    # Define paths
    model_path = 'hydrolase_models/your_dataset_ESM2-650_randomid_inference.pth'
    trembl_fasta = 'path/to/trembl.fasta'
    output_path = 'hydrolase_predictions.csv'
    embedding_path = 'embeddings/ESM2-650_False.pth'  # Pre-computed embeddings
    
    # Run screening (simple one-liner)
    results = run_trembl_screening(
        model_path=model_path,
        trembl_fasta=trembl_fasta,
        output_path=output_path,
        embedding_path=embedding_path,
        confidence_threshold=0.8,
        max_sequences=10000  # Limit for testing
    )
    
    print(f"Screening completed! Found {results['positive_count']} potential hydrolases.")


def screen_trembl_advanced_example():
    """Advanced example with custom settings."""
    
    # Set up inference arguments
    args = InferenceArguments(
        model_path='hydrolase_models/your_dataset_ESM2-650_randomid_inference.pth',
        embedding_path='embeddings/ESM2-650_False.pth',
        batch_size=128,  # Larger batch for faster processing
        confidence_threshold=0.8,
        save_probabilities=True,
        output_format='csv',
        device='cuda'  # Use GPU if available
    )
    
    # Initialize inference pipeline
    inference = HydrolaseInference(args)
    
    # Load TrEMBL sequences (you can also load from a list)
    results = inference.predict_from_fasta(
        'path/to/trembl.fasta',
        max_sequences=50000  # Process 50k sequences
    )
    
    # Save results
    inference.save_predictions(results, 'hydrolase_predictions_advanced.csv')
    
    # Filter high-confidence predictions
    high_confidence_seqs = []
    for i, (seq_id, seq, pred, prob) in enumerate(zip(
        results['sequence_ids'], 
        results['sequences'],
        results['predictions'], 
        results['probabilities']
    )):
        if pred == 1 and prob[1] > 0.9:  # Very high confidence hydrolases
            high_confidence_seqs.append({
                'id': seq_id,
                'sequence': seq,
                'confidence': prob[1]
            })
    
    print(f"Found {len(high_confidence_seqs)} high-confidence hydrolases")
    
    # Save high-confidence sequences to FASTA
    with open('high_confidence_hydrolases.fasta', 'w') as f:
        for seq_data in high_confidence_seqs:
            f.write(f">{seq_data['id']} confidence={seq_data['confidence']:.3f}\n")
            f.write(f"{seq_data['sequence']}\n")


def screen_custom_sequences():
    """Example of screening custom protein sequences."""
    
    # Custom sequences to test
    sequences = [
        "MKTLLITGLLLGTTVLMSTQSLLLKWLQPLLSQCQSLIRQNVS",  # Example sequence 1
        "MALTAAMFKKKHQGPGKGPMLVGLKQAWSRPTSGPVGPVEIKGGVKLTNVNT",  # Example sequence 2
        # Add more sequences...
    ]
    
    sequence_ids = ['custom_seq_1', 'custom_seq_2']
    
    # Set up inference
    args = InferenceArguments(
        model_path='hydrolase_models/your_dataset_ESM2-650_randomid_inference.pth',
        embedding_model='ESM2-650',  # Generate embeddings on-the-fly
        confidence_threshold=0.7
    )
    
    inference = HydrolaseInference(args)
    
    # Run prediction
    results = inference.predict(sequences, sequence_ids)
    
    # Print results
    for seq_id, prediction, probability in zip(
        results['sequence_ids'], 
        results['predictions'], 
        results['probabilities']
    ):
        confidence = probability[1] if prediction == 1 else probability[0]
        label = "Hydrolase" if prediction == 1 else "Non-hydrolase"
        print(f"{seq_id}: {label} (confidence: {confidence:.3f})")


def batch_processing_large_datasets():
    """Example for processing very large datasets efficiently."""
    
    # For very large datasets, use SQL storage to avoid memory issues
    args = InferenceArguments(
        model_path='hydrolase_models/your_dataset_ESM2-650_randomid_inference.pth',
        embedding_path='embeddings/ESM2-650_False.db',  # SQL database
        use_sql=True,
        batch_size=256,  # Large batch size
        num_workers=4,   # Multi-processing
        confidence_threshold=0.8
    )
    
    inference = HydrolaseInference(args)
    
    # Process in chunks to manage memory
    chunk_size = 100000  # Process 100k sequences at a time
    total_positives = 0
    
    # This is a conceptual example - you'd implement actual chunking logic
    for chunk_start in range(0, 1000000, chunk_size):  # 1M sequences total
        print(f"Processing chunk {chunk_start//chunk_size + 1}...")
        
        # Load chunk of sequences (implement your loading logic here)
        chunk_sequences = load_sequence_chunk(chunk_start, chunk_size)
        
        # Run inference on chunk
        results = inference.predict(chunk_sequences)
        
        # Save chunk results
        chunk_output = f'hydrolase_predictions_chunk_{chunk_start//chunk_size + 1}.csv'
        inference.save_predictions(results, chunk_output)
        
        total_positives += results['positive_count']
        print(f"Chunk {chunk_start//chunk_size + 1}: {results['positive_count']} positives")
    
    print(f"Total hydrolases found: {total_positives}")


def load_sequence_chunk(start_idx, chunk_size):
    """Dummy function - implement your own sequence loading logic."""
    # This would load a chunk of sequences from your large dataset
    # Could read from FASTA, database, etc.
    return [f"SEQUENCE_{i}" for i in range(start_idx, start_idx + chunk_size)]


if __name__ == "__main__":
    # Choose which example to run
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_example.py <mode>")
        print("Modes: train, screen_simple, screen_advanced, custom, batch")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "train":
        train_hydrolase_model()
    elif mode == "screen_simple":
        screen_trembl_simple_example()
    elif mode == "screen_advanced":
        screen_trembl_advanced_example()
    elif mode == "custom":
        screen_custom_sequences()
    elif mode == "batch":
        batch_processing_large_datasets()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: train, screen_simple, screen_advanced, custom, batch") 