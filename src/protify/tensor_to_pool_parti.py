import os
import torch
import esm
import re
from Bio import SeqIO
import networkx as nx
import numpy as np
import gc

def parse_fasta(fasta_file):
    """Parses a FASTA file, handling potential issues and duplicate IDs.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        list: A list of (label, sequence) tuples.  Labels are made unique.
                Returns an empty list and prints an error if the file
                cannot be parsed.
    """
    data = []
    try:
        records = list(SeqIO.parse(fasta_file, "fasta"))
    except Exception as e:
        print(f"Error parsing FASTA file: {e}")
        return []

    seen_ids = {}
    for record in records:
        sequence = str(record.seq).upper()  # Ensure uppercase
        label = record.id.strip()  # Remove leading/trailing spaces
        if not label:
            label = "unnamed_sequence"  # Default label for empty IDs

        # Make IDs unique
        if label in seen_ids:
            seen_ids[label] += 1
            label = f"{label}_{seen_ids[label]}"  # Add a counter
        else:
            seen_ids[label] = 1

        # Basic sequence cleaning (remove non-amino acid chars)
        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", sequence)
        data.append((label, sequence))
    return data

class TokenToSequencePooler:
    def __init__(self, token_emb, attention_layers):
        # Initialize the pooler by loading token embeddings and attention layers from the given paths.
        # Handles the removal of CLS and END token embeddings from representations.

        self.token_emb = token_emb
        self.attention_layers = attention_layers
        if self.token_emb is not None:
          if len(self.token_emb.shape) == 2:
            self.representations =  self.token_emb[1:-1]
          elif len(self.token_emb.shape) == 3:
            self.representations = self.token_emb[:,1:-1, :]
        else:
          self.representations = None

        self.attn_all_layers = self.attention_layers

    # def cls_pooling(self, save_path=None):
    #     # Extract the CLS token from the representations with CLS.
    #     # Optionally save the CLS token to the specified path.
    #     # If representations are not loaded, handle the error and return None.

    #     if self.token_emb is not None:
    #         cls_token = self.token_emb[0]
    #         if save_path:
    #             torch.save(save_path, cls_token)
    #         return cls_token.squeeze()
    #     print(f"token_emb was None for sequence ", flush=True)
    #     return None  # Handle cases where CLS token is not available or representations are not loaded properly

    def create_pooled_matrices_across_layers(self, mtx_all_layers):
        # Perform max pooling across layers by selecting the maximum values across attention layers.
        # Returns the matrix after pooling the attention layers.

        mtx_max_of_max = torch.max(mtx_all_layers[1], dim=1)[0]
        return mtx_max_of_max

    def mean_pooling(self):
        # Perform mean pooling on the token representations by averaging across all tokens.

        if self.representations is not None:
            if len(self.representations.shape) == 2:
                return np.mean(self.representations, axis=0)
            else:
                return np.mean(self.representations, axis=1)
        return None

    def max_pooling(self):
        # Perform max pooling on the token representations.

        if self.representations is not None:
            if len(self.representations.shape) == 2:
                return np.max(self.representations, axis=0)
            else:
                return np.max(self.representations, axis=1)
        return None

    def pool_parti(self,
                  verbose=False,
                  return_importance=False):
        # Perform pooling based on PageRank algorithm applied to attention matrices.
        # Optionally return importance weights or print details about the importance calculation.
        # Handles errors during the pooling process by printing detailed information.

        matrix_to_pool = self.create_pooled_matrices_across_layers(mtx_all_layers=self.attn_all_layers).squeeze().numpy()

        dict_importance = self._page_rank(matrix_to_pool)
        importance_weights = np.array(list(self._calculate_importance_weights(dict_importance).values()))

        if return_importance:
            return importance_weights
        if verbose:
            print(f'pagerank direct outcome is {dict_importance}\n')
            print(f'importance_weights dict of length {len(importance_weights)} looks like\n {importance_weights}')
            print(f'shape of the importance matrix is {len(importance_weights)} and for repr, its {self.representations.shape}')
            print(f"importance weights look like {sorted(importance_weights, reverse=True)[0:5]}")

        try:
            return torch.tensor(np.average(self.representations, weights=importance_weights, axis=0))
        except Exception as e:
            print(f"{e} in PageRank without cls", flush=True)
            print(f"self.representations shape {self.representations.shape}", flush=True)
            print(f"importance_weights {len(importance_weights)}", flush=True)
            return None

    def _page_rank(self, attention_matrix, personalization=None, nstart=None, prune_type="top_k_outdegree"):
        # Run PageRank on the attention matrix converted to a graph.
        # Raises exceptions if the graph doesn't match the token sequence or has no edges.
        # Returns the PageRank scores for each token node.

        G = self._convert_to_graph(attention_matrix)

        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(
                f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.")
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.")

        return nx.pagerank(G, alpha=0.85, tol=1e-06, weight='weight', personalization=personalization,
                           nstart=nstart, max_iter=100)

    def _convert_to_graph(self, matrix):
        # Convert a matrix (e.g., attention scores) to a directed graph using networkx.
        # Each element in the matrix represents a directed edge with a weight.

        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)

        return G

    def _calculate_importance_weights(self, dict_importance):
        # Normalize the PageRank scores (importance values) so they sum to 1.
        # Exclude CLS and END token from the importance calculation.

        # Get the highest integer key
        highest_key = max(dict_importance.keys())

        # Remove the entry with the highest key (END) and the entry with key 0 (CLS token)
        del dict_importance[highest_key]
        del dict_importance[0]

        total = sum(dict_importance.values())
        return {k: v / total for k, v in dict_importance.items()}

def main_pooling(token_emb, 
                 attention_layers, 
                #  output_dir, 
                #  generate_all
                 ):
    """
    Main function to perform pooling operations on protein sequence data.

    Args:
        token_emb (str): Path to the token embeddings file.
        attention_layers (str): Path to the attention matrices file.
        output_dir (str): Directory where the output embeddings will be saved.
        generate_all (bool): If True, generates all pooling embeddings (CLS, mean, max, Pool PaRTI).
                                        If False, only generates the Pool PaRTI embedding.
    """
    # # Create the output directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)
    # file_name = os.path.basename(token_emb)

    # Instantiate the TokenToSequencePooler
    pooler = TokenToSequencePooler(token_emb=token_emb,
                                    attention_layers=attention_layers)
    if pooler.token_emb is None or pooler.attn_all_layers is None:
        print(f"Skipping pooling due to missing data.")
        return

    rep_w_cls = pooler.token_emb
    attn = pooler.attn_all_layers

    # Check if the shapes of representations and attentions match
    if not rep_w_cls.shape[0] == attn.shape[-1]:
        if len(rep_w_cls.shape) == 3 and not rep_w_cls.shape[1] == attn.shape[-1]:
            print(f"The attention and representation shapes don't match", flush=True)
            return

    # Perform Pool PaRTI pooling
    # pool_parti_dir = os.path.join(output_dir, "pool_parti")
    # os.makedirs(pool_parti_dir, exist_ok=True)
    # address = os.path.join(pool_parti_dir, file_name)
    pooled = pooler.pool_parti(verbose=False, return_importance=False)
    return pooled
    #     if pooled is not None:
    #         torch.save(pooled, address)
    #         print(f"Pool PaRTI embedding saved at {address}")
    #     else:
    #         print(f"Pool PaRTI pooling failed")
    # else:
    #     print(f"Pool PaRTI embedding already exists at {address}")

    # # If generate_all is True, perform additional pooling methods
    # if generate_all:
    #     # CLS Pooling
    #     cls_pooled_dir = os.path.join(output_dir, "cls_pooled")
    #     os.makedirs(cls_pooled_dir, exist_ok=True)
    #     address = os.path.join(cls_pooled_dir, file_name)
    #     if not os.path.exists(address):
    #         cls_pooled = pooler.cls_pooling()
    #         if cls_pooled is not None:
    #             torch.save(cls_pooled, address)
    #             print(f"CLS-pooled embedding saved at {address}")
    #         else:
    #             print(f"CLS pooling failed for {file_name}, skipping save.")
    #     else:
    #         print(f"CLS-pooled embedding already exists at {address}")

    #     # Mean Pooling
    #     mean_pooled_dir = os.path.join(output_dir, "mean_pooled")
    #     os.makedirs(mean_pooled_dir, exist_ok=True)
    #     address = os.path.join(mean_pooled_dir, file_name)
    #     if not os.path.exists(address):
    #         mean_pooled = pooler.mean_pooling()
    #         if mean_pooled is not None:
    #             torch.save(mean_pooled, address)
    #             print(f"Mean-pooled embedding saved at {address}")
    #         else:
    #             print(f"Mean pooling failed for {file_name}, skipping save.")
    #     else:
    #         print(f"Mean-pooled embedding already exists at {address}")

    #     # Max Pooling
    #     max_pooled_dir = os.path.join(output_dir, "max_pooled")
    #     os.makedirs(max_pooled_dir, exist_ok=True)
    #     address = os.path.join(max_pooled_dir, file_name)
    #     if not os.path.exists(address):
    #         max_pooled = pooler.max_pooling()
    #         if max_pooled is not None:
    #             torch.save(max_pooled, address)
    #             print(f"Max-pooled embedding saved at {address}")
    #         else:
    #             print(f"Max pooling failed for {file_name}, skipping save.")
    #     else:
    #         print(f"Max-pooled embedding already exists at {address}")

    print(f"Pooling operations completed for {file_name}.")


def process_fasta_and_extract_data(fasta_file, output_dir, batch_size=1, max_seq_len=10000, use_gpu=True):
    """
    Process a FASTA file and extract ESM-2 data, then perform pooling, with memory-efficient batching.

    Args:
        fasta_file: Path to the FASTA file
        output_dir: Directory to save outputs
        batch_size: Number of sequences to process at once
        max_seq_len: Maximum sequence length to consider (sequences will be truncated)
        use_gpu: Whether to use GPU acceleration if available
    """
    # Create output directories
    os.makedirs(f"{output_dir}/attention_matrices_mean_max_perLayer", exist_ok=True)
    os.makedirs(f"{output_dir}/representation_matrices", exist_ok=True)
    os.makedirs(f"{output_dir}/pooled_embeddings", exist_ok=True)  # Directory for pooled embeddings

    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results

    # Use GPU if available and requested
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    # Read all sequences from the FASTA file
    all_data = parse_fasta(fasta_file)

    # Process in batches
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch_data)} sequences)")

        # Process each sequence in the batch
        batch_to_process = []
        for idx, (label, seq) in enumerate(batch_data):
            # Truncate sequence if too long
            if len(seq) > max_seq_len:
                print(f"Warning: Sequence {label} truncated from {len(seq)} to {max_seq_len}")
                seq = seq[:max_seq_len]

            # Skip sequences that are too short
            if len(seq) < 2:
                print(f"Warning: Sequence {label} is too short ({len(seq)} amino acids), skipping")
                continue

            # Check if output files already exist for this sequence
            base_name = os.path.basename(fasta_file).split('.fa')[0].split('.fasta')[0]
            seq_id = f"{base_name}_{label}" if len(all_data) > 1 else base_name

            attention_file_path = f"{output_dir}/attention_matrices_mean_max_perLayer/{seq_id}.pt"
            representations_file_path = f"{output_dir}/representation_matrices/{seq_id}.pt"
            pooled_embedding_path = f"{output_dir}/pooled_embeddings/{seq_id}.pt"  # Path for pooled embedding

            # Skip if all files already exist
            if os.path.exists(attention_file_path) and os.path.exists(
                    representations_file_path) and os.path.exists(pooled_embedding_path):
                print(f"Skipping already processed sequence: {label}")
                continue

            batch_to_process.append((label, seq))

        # Skip the batch if all sequences are already processed
        if not batch_to_process:
            print("All sequences in this batch already processed, skipping")
            continue

        try:
            # Convert batch data
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_to_process)
            batch_tokens = batch_tokens.to(device)

            # Extract per-residue representations, contacts, and attention heads
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)

            # Process each sequence in the batch
            for j, (label, _) in enumerate(batch_to_process):
                # Get sequence ID
                base_name = os.path.basename(fasta_file).split('.fa')[0].split('.fasta')[0]
                seq_id = f"{base_name}_{label}" if len(all_data) > 1 else base_name

                attention_file_path = f"{output_dir}/attention_matrices_mean_max_perLayer/{seq_id}.pt"
                representations_file_path = f"{output_dir}/representation_matrices/{seq_id}.pt"
                pooled_embedding_path = f"{output_dir}/pooled_embeddings/{seq_id}.pt"  # Path for pooled embedding

                # Process and save attention heads across layers
                if not os.path.exists(attention_file_path):
                    attn_mean_pooled_layers = []
                    attn_max_pooled_layers = []

                    for layer in range(33):
                        attn_raw = results["attentions"][j, layer].cpu()  # Move to CPU for processing

                        # Compress attention data
                        attn_mean_pooled = torch.mean(attn_raw, dim=0)
                        attn_max_pooled = torch.max(attn_raw, dim=0).values

                        attn_mean_pooled_layers.append(attn_mean_pooled)
                        attn_max_pooled_layers.append(attn_max_pooled)

                    # Stack the pooled attention matrices
                    attn_mean_pooled_stacked = torch.stack(attn_mean_pooled_layers)
                    attn_max_pooled_stacked = torch.stack(attn_max_pooled_layers)
                    combined_attention = torch.stack([attn_mean_pooled_stacked, attn_max_pooled_stacked]).unsqueeze(1)

                    try:
                        torch.save(combined_attention, attention_file_path)
                        print(f"Saved attention data: {attention_file_path}")
                    except Exception as e:
                        print(f"Error saving attention data for {seq_id}: {e}")

                # Save representations
                if not os.path.exists(representations_file_path):
                    representations = results["representations"][33][j].cpu()  # Move to CPU for saving
                    torch.save(representations, representations_file_path)
                    print(f"Saved representations: {representations_file_path}")
                
                # Perform Pooling and save
                if not os.path.exists(pooled_embedding_path):
                  try:
                    # Instantiate the TokenToSequencePooler
                    pooler = TokenToSequencePooler(token_emb=representations_file_path,
                                                    attention_layers=attention_file_path)
                    pooled_embedding = pooler.pool_parti(verbose=False, return_importance=False) # Perform the pooling

                    if pooled_embedding is not None:
                        torch.save(pooled_embedding, pooled_embedding_path)
                        print(f"Pooled embedding saved at {pooled_embedding_path}")
                    else:
                        print(f"Pooling failed for {seq_id}, skipping save.")
                  except Exception as e:
                        print(f"Error during pooling for {seq_id}: {e}")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"GPU out of memory error. Reducing batch size for this batch.")
                # Clear cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()

                # Process one by one as fallback
                for (label, seq) in batch_to_process:
                    print(f"Processing individual sequence: {label}")
                    try:
                        process_single_sequence(model, alphabet, label, seq, fasta_file, output_dir, device)
                    except Exception as inner_e:
                        print(f"Failed to process sequence {label}: {inner_e}")
            else:
                print(f"Error processing batch: {e}")

        # Clear memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
def process_single_sequence(model, alphabet, label, seq, fasta_file, output_dir, device):
    """Process a single sequence when batch processing fails."""
    batch_converter = alphabet.get_batch_converter()

    # Prepare single sequence
    data = [(label, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Generate output paths
    base_name = os.path.basename(fasta_file).split('.fa')[0].split('.fasta')[0]
    seq_id = f"{base_name}_{label}"

    attention_file_path = f"{output_dir}/attention_matrices_mean_max_perLayer/{seq_id}.pt"
    representations_file_path = f"{output_dir}/representation_matrices/{seq_id}.pt"
    pooled_embedding_path = f"{output_dir}/pooled_embeddings/{seq_id}.pt"

    # Skip if all already processed
    if os.path.exists(attention_file_path) and os.path.exists(
            representations_file_path) and os.path.exists(pooled_embedding_path):
        return

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Process and save attention
    if not os.path.exists(attention_file_path):
        attn_mean_pooled_layers = []
        attn_max_pooled_layers = []

        for layer in range(33):
            attn_raw = results["attentions"][0, layer].cpu()
            attn_mean_pooled = torch.mean(attn_raw, dim=0)
            attn_max_pooled = torch.max(attn_raw, dim=0).values

            attn_mean_pooled_layers.append(attn_mean_pooled)
            attn_max_pooled_layers.append(attn_max_pooled)

        attn_mean_pooled_stacked = torch.stack(attn_mean_pooled_layers)
        attn_max_pooled_stacked = torch.stack(attn_max_pooled_layers)
        combined_attention = torch.stack([attn_mean_pooled_stacked, attn_max_pooled_stacked]).unsqueeze(1)

        torch.save(combined_attention, attention_file_path)
        print(f"Saved attention data: {attention_file_path}")

    # Save representations
    if not os.path.exists(representations_file_path):
        representations = results["representations"][33][0].cpu()
        torch.save(representations, representations_file_path)
        print(f"Saved representations: {representations_file_path}")
    
    # Perform Pooling and Save
    if not os.path.exists(pooled_embedding_path):
        try:
          pooler = TokenToSequencePooler(path_token_emb=representations_file_path,
                                        path_attention_layers=attention_file_path)
          pooled_embedding = pooler.pool_parti(verbose=False, return_importance=False)
          if pooled_embedding is not None:
            torch.save(pooled_embedding, pooled_embedding_path)
            print(f"Pooled embedding saved at {pooled_embedding_path}")
        except Exception as e:
            print(f"Error during pooling: {e}")

## Arguments
fasta_file = r''  # Path to FASTA file
output_dir = r''  # Path to output directory
batch_size = 1  # It is reccomended to use a small batch size to avoid too much GPU use
max_seq_len = 10000  # Set a maximum length for a sequence to be extracted from the FASTA file. Will be truncated to this length if a sequence has more AA than the value provided.
use_gpu = True  # Choose to use GPU or not


process_fasta_and_extract_data(fasta_file=fasta_file, output_dir=output_dir, batch_size=batch_size, max_seq_len=max_seq_len, use_gpu=use_gpu)

