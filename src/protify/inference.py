import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm.auto import tqdm

from protify.probes.linear_probe import LinearProbe
from protify.probes.transformer_probe import TransformerProbe
from protify.data.dataset_classes import InferenceDataset, InferenceDatasetFromDisk
from protify.data.data_collators import InferenceCollator
from embedder import Embedder, EmbeddingArguments
from protify.utils import torch_load, print_message
from protify.probes.linear_probe import LinearProbeConfig


@dataclass
class InferenceArguments:
    """Arguments for inference configuration."""
    def __init__(
            self,
            model_path: str,
            embedding_path: Optional[str] = None,
            batch_size: int = 64,
            num_workers: int = 0,
            device: str = 'auto',
            output_format: str = 'csv',  # 'csv', 'json', 'tsv'
            confidence_threshold: float = 0.5,
            save_probabilities: bool = True,
            use_sql: bool = False,
            embedding_model: Optional[str] = None,
            **kwargs
    ):
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_format = output_format
        self.confidence_threshold = confidence_threshold
        self.save_probabilities = save_probabilities
        self.use_sql = use_sql
        self.embedding_model = embedding_model


class HydrolaseInference:
    """
    Class for conducting inference on protein sequences using trained hydrolase models.
    
    Example usage:
        # Initialize inference pipeline
        inference = HydrolaseInference(
            model_path='path/to/trained_model.pth',
            embedding_path='path/to/embeddings.pth'
        )
        
        # Load TrEMBL sequences
        sequences = load_trembl_sequences()
        
        # Run inference
        predictions = inference.predict(sequences)
        
        # Save results
        inference.save_predictions(predictions, 'hydrolase_predictions.csv')
    """
    
    def __init__(self, args: InferenceArguments):
        self.args = args
        self.device = torch.device(args.device)
        self.model = None
        self.embeddings_dict = None
        
        print_message(f'Initialized inference pipeline on device: {self.device}')
        
        # Load the trained model
        self._load_model()
        
    def _load_model(self):
        """Load the trained probe model."""
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model not found: {self.args.model_path}")
            
        print_message(f'Loading model from {self.args.model_path}')
        
        # Try to load as a complete model first
        try:
            self.model = torch.load(self.args.model_path, map_location=self.device)
        except:
            # If that fails, try loading state dict
            checkpoint = torch_load(self.args.model_path)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                config = checkpoint.get('config', None)
            else:
                state_dict = checkpoint
                config = None
                
            # You'll need to specify the correct model type and config
            # This should match what you used during training
            if config:
                self.model = LinearProbe(config)
            else:
                # Default config - you may need to adjust these parameters
                config = LinearProbeConfig(
                    input_dim=768,  # Adjust based on your embedding model
                    hidden_dim=8192,
                    num_labels=2,  # Binary classification
                    task_type='singlelabel'
                )
                self.model = LinearProbe(config)
                
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        print_message('Model loaded successfully')
        
    def _load_embeddings(self, sequences: List[str]) -> Union[Dict, str]:
        """Load or generate embeddings for the given sequences."""
        if self.args.embedding_path and os.path.exists(self.args.embedding_path):
            if self.args.use_sql:
                print_message(f'Using SQL embeddings from {self.args.embedding_path}')
                return self.args.embedding_path
            else:
                print_message(f'Loading embeddings from {self.args.embedding_path}')
                embeddings_dict = torch_load(self.args.embedding_path)
                
                # Check if all sequences have embeddings
                missing_seqs = [seq for seq in sequences if seq not in embeddings_dict]
                if missing_seqs:
                    print_message(f'Missing embeddings for {len(missing_seqs)} sequences')
                    if self.args.embedding_model:
                        print_message('Generating missing embeddings...')
                        missing_embeddings = self._generate_embeddings(missing_seqs)
                        embeddings_dict.update(missing_embeddings)
                    else:
                        raise ValueError(f'Missing embeddings for {len(missing_seqs)} sequences and no embedding model specified')
                        
                return embeddings_dict
        else:
            if not self.args.embedding_model:
                raise ValueError('No embeddings found and no embedding model specified')
            print_message('Generating embeddings for all sequences...')
            return self._generate_embeddings(sequences)
            
    def _generate_embeddings(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Generate embeddings for sequences using the specified model."""
        if not self.args.embedding_model:
            raise ValueError('No embedding model specified')
            
        # Set up embedding arguments
        embedding_args = EmbeddingArguments(
            embedding_batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            save_embeddings=False,
            sql=False
        )
        
        # Generate embeddings
        embedder = Embedder(embedding_args, sequences)
        embeddings_dict = embedder(self.args.embedding_model)
        
        return embeddings_dict
        
    def predict(self, sequences: List[str], sequence_ids: Optional[List[str]] = None) -> Dict:
        """
        Run inference on a list of protein sequences.
        
        Args:
            sequences: List of protein sequences
            sequence_ids: Optional list of sequence IDs (defaults to indices)
            
        Returns:
            Dictionary containing predictions, probabilities, and metadata
        """
        if sequence_ids is None:
            sequence_ids = [f'seq_{i}' for i in range(len(sequences))]
            
        print_message(f'Running inference on {len(sequences)} sequences')
        
        # Load/generate embeddings
        embeddings = self._load_embeddings(sequences)
        
        # Create dataset
        if self.args.use_sql:
            dataset = InferenceDatasetFromDisk(
                sequences=sequences,
                db_path=embeddings,
                batch_size=self.args.batch_size
            )
        else:
            dataset = InferenceDataset(
                sequences=sequences,
                emb_dict=embeddings
            )
            
        # Create data loader
        collator = InferenceCollator(full=False)  # Adjust based on your embedding type
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collator,
            shuffle=False
        )
        
        # Run inference
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Running inference'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Convert to probabilities
                if self.model.config.task_type == 'regression':
                    # For regression, logits are the direct predictions
                    probabilities = torch.sigmoid(logits)  # Convert to [0,1] range
                    predictions = (probabilities > self.args.confidence_threshold).long()
                else:
                    # For classification, apply softmax
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1)
                    
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        print_message(f'Inference completed. Found {sum(all_predictions)} positive predictions')
        
        # Compile results
        results = {
            'sequence_ids': sequence_ids,
            'sequences': sequences,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'positive_count': sum(all_predictions),
            'total_count': len(all_predictions)
        }
        
        return results
        
    def predict_from_fasta(self, fasta_path: str, max_sequences: Optional[int] = None) -> Dict:
        """
        Run inference on sequences from a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            max_sequences: Maximum number of sequences to process (for testing)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        sequences, sequence_ids = self._load_fasta(fasta_path, max_sequences)
        return self.predict(sequences, sequence_ids)
        
    def _load_fasta(self, fasta_path: str, max_sequences: Optional[int] = None) -> tuple:
        """Load sequences from a FASTA file."""
        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("Biopython is required for FASTA file processing. Install with: pip install biopython")
            
        sequences = []
        sequence_ids = []
        
        print_message(f'Loading sequences from {fasta_path}')
        
        with open(fasta_path, 'r') as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                if max_sequences and i >= max_sequences:
                    break
                    
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
                
        print_message(f'Loaded {len(sequences)} sequences from FASTA file')
        return sequences, sequence_ids
        
    def save_predictions(self, results: Dict, output_path: str):
        """Save prediction results to file."""
        # Create DataFrame
        df_data = {
            'sequence_id': results['sequence_ids'],
            'sequence': results['sequences'],
            'prediction': results['predictions']
        }
        
        if self.args.save_probabilities:
            if len(results['probabilities'][0]) > 1:
                # Multi-class probabilities
                for i in range(len(results['probabilities'][0])):
                    df_data[f'probability_class_{i}'] = [prob[i] for prob in results['probabilities']]
            else:
                # Binary classification - single probability
                df_data['probability'] = [prob[0] if isinstance(prob, (list, np.ndarray)) else prob 
                                        for prob in results['probabilities']]
                
        df = pd.DataFrame(df_data)
        
        # Add metadata columns
        df['is_hydrolase'] = df['prediction'] == 1
        
        # Save based on format
        if self.args.output_format == 'csv':
            df.to_csv(output_path, index=False)
        elif self.args.output_format == 'tsv':
            df.to_csv(output_path, sep='\t', index=False)
        elif self.args.output_format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {self.args.output_format}")
            
        print_message(f'Results saved to {output_path}')
        print_message(f'Summary: {results["positive_count"]}/{results["total_count"]} sequences predicted as hydrolases')


def run_trembl_screening(
        model_path: str,
        trembl_fasta: str,
        output_path: str,
        embedding_path: Optional[str] = None,
        embedding_model: str = 'ESM2-650',
        batch_size: int = 64,
        confidence_threshold: float = 0.8,
        max_sequences: Optional[int] = None
    ) -> Dict:
    """
    Convenience function to screen TrEMBL for hydrolases.
    
    Args:
        model_path: Path to trained model
        trembl_fasta: Path to TrEMBL FASTA file
        output_path: Path to save results
        embedding_path: Path to pre-computed embeddings (optional)
        embedding_model: Model to use for generating embeddings
        batch_size: Batch size for inference
        confidence_threshold: Threshold for positive predictions
        max_sequences: Maximum sequences to process (for testing)
        
    Returns:
        Dictionary with prediction results
    """
    args = InferenceArguments(
        model_path=model_path,
        embedding_path=embedding_path,
        embedding_model=embedding_model,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )
    
    inference = HydrolaseInference(args)
    results = inference.predict_from_fasta(trembl_fasta, max_sequences)
    inference.save_predictions(results, output_path)
    
    return results 