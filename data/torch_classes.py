### imports
import random
import torch
import numpy as np
import sqlite3
import torch.nn.functional as F
from typing import List, Tuple
from tqdm.auto import tqdm
from torch.utils.data import Dataset as TorchDataset
from .utils import pad_and_concatenate_dimer
# from torch.nn.utils.rnn import pad_sequence


def _pad_matrix_embeds(embeds: List[torch.Tensor], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # pad and concatenate, return padded embeds and mask
    padded_embeds, attention_masks = [], []
    for embed in embeds:
        seq_len = embed.size(0)
        padding_size = max_len - seq_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(max_len, dtype=torch.long)
        if padding_size > 0:
            attention_mask[seq_len:] = 0
            
            # Pad along the sequence dimension (dim=0)
            padding = torch.zeros((padding_size, embed.size(1)), dtype=embed.dtype)
            padded_embed = torch.cat((embed, padding), dim=0)
        else:
            padded_embed = embed
            
        padded_embeds.append(padded_embed)
        attention_masks.append(attention_mask)
        
    return torch.stack(padded_embeds), torch.stack(attention_masks)


def string_labels_collator_builder(tokenizer, **kwargs):
    def _collate_fn(batch):
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([torch.tensor(ex[1]) for ex in batch])
        batch = tokenizer(seqs,
                          padding='longest',
                          padding_to_multiple_of=8,
                          truncation=False,
                          return_tensors='pt',
                          add_special_tokens=True)
        batch['labels'] = labels
        return batch
    return _collate_fn


def embeds_labels_collator_builder(full=False, task_type='tokenwise', **kwargs):
    def _collate_fn(batch):
        if full:
            embeds = [ex[0] for ex in batch]
            labels = [ex[1] for ex in batch]
            
            # Find max sequence length for padding
            max_len = max(embed.size(0) for embed in embeds)
            
            embeds, attention_mask = _pad_matrix_embeds(embeds, max_len)
            
            # Pad labels
            if task_type == 'tokenwise':
                padded_labels = []
                for label in labels:
                    padding_size = max_len - label.size(0)
                    if padding_size > 0:
                        # Use -100 as padding value for labels (ignored by loss functions)
                        padding = torch.full((padding_size,), -100, dtype=label.dtype)
                        padded_label = torch.cat((label.squeeze(-1), padding))
                    else:
                        padded_label = label.squeeze(-1)
                    padded_labels.append(padded_label)
            
            labels = torch.stack(padded_labels)
            
            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:
            embeds = torch.stack([ex[0] for ex in batch])
            labels = torch.stack([ex[1] for ex in batch])
        
            return {
                'embeddings': embeds,
                'labels': labels
            }
    return _collate_fn


def pair_string_collator_builder(tokenizer, **kwargs):
    def _collate_fn(batch):
        seqa = [f[0] for f in batch]
        seqb = [f[1] for f in batch]
        labels = torch.stack([torch.tensor(f[2]) for f in batch])
        a = tokenizer(seqa,
                      padding='longest',
                      padding_to_multiple_of=8,
                      truncation=False,
                      return_tensors='pt',
                      add_special_tokens=True)
        b = tokenizer(seqb, 
                      padding='longest',
                      padding_to_multiple_of=8,
                      truncation=False,
                      return_tensors='pt',
                      add_special_tokens=True)
        return {
            'a_tokenized': a,
            'b_tokenized': b,
            'labels': labels
        }
    return _collate_fn


def pair_embeds_labels_collator_builder(full=False, **kwargs):
    def _collate_fn(batch):
        if full:
            embeds_a = [ex[0] for ex in batch]
            embeds_b = [ex[1] for ex in batch]
            max_len_a = max(embed.size(0) for embed in embeds_a)
            max_len_b = max(embed.size(0) for embed in embeds_b)
            embeds_a, attention_mask_a = _pad_matrix_embeds(embeds_a, max_len_a)
            embeds_b, attention_mask_b = _pad_matrix_embeds(embeds_b, max_len_b)
            embeds, attention_mask = pad_and_concatenate_dimer(embeds_a, embeds_b, attention_mask_a, attention_mask_b)

            labels = torch.stack([ex[2] for ex in batch])

            return {
                'embeddings': embeds,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            embeds_a = torch.stack([ex[0] for ex in batch])
            embeds_b = torch.stack([ex[1] for ex in batch]) 
            labels = torch.stack([ex[2] for ex in batch])
            embeds = torch.cat([embeds_a, embeds_b], dim=-1)
        return {
            'embeddings': embeds,
            'labels': labels
        }
    return _collate_fn


class PairEmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_a='SeqA',
            col_b='SeqB',
            label_col='labels',
            full=False, 
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=1000,
            input_dim=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a, self.seqs_b, self.labels = hf_dataset[col_a], hf_dataset[col_b], hf_dataset[label_col]
        self.db_file = db_path
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.full = full
        self.length = len(self.labels)
        self.read_amt = read_scaler * self.batch_size
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0
        self.task_type = task_type

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        missing_seqs = [seq for seq in self.seqs_a + self.seqs_b if seq not in all_seqs]
        if missing_seqs:
            print('Sequences not found in embeddings:', missing_seqs)
        else:
            print('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs_a, self.seqs_b, self.labels))
        random.shuffle(data)
        self.seqs_a, self.seqs_b, self.labels = zip(*data)
        self.seqs_a, self.seqs_b, self.labels = list(self.seqs_a), list(self.seqs_b), list(self.labels)
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0

    def get_embedding(self, c, seq):
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        if row is None:
            raise ValueError(f"Embedding not found for sequence: {seq}")
        emb_data = row[0]
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_dim))
        return emb

    def read_embeddings(self):
        embeddings_a, embeddings_b, labels = [], [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            emb_a = self.get_embedding(c, self.seqs_a[i])
            emb_b = self.get_embedding(c, self.seqs_b[i])
            embeddings_a.append(emb_a)
            embeddings_b.append(emb_b)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb_a = self.embeddings_a[self.index]
        emb_b = self.embeddings_b[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb_a, emb_b, label


class PairEmbedsLabelsDataset(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            emb_dict,
            col_a='SeqA',
            col_b='SeqB',
            full=False,
            label_col='labels',
            input_dim=768,
            task_type='regression',
            **kwargs
        ):
        self.seqs_a = hf_dataset[col_a]
        self.seqs_b = hf_dataset[col_b]
        self.labels = hf_dataset[label_col]
        self.input_dim = input_dim // 2 if not full else input_dim # already scaled if ppi
        self.task_type = task_type
        self.full = full

        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(hf_dataset[col_a] + hf_dataset[col_b])
        # Filter emb_dict to keep only the necessary embeddings
        self.emb_dict = {seq: emb_dict[seq] for seq in needed_seqs if seq in emb_dict}
        # Check for any missing embeddings
        missing_seqs = needed_seqs - self.emb_dict.keys()
        if missing_seqs:
            raise ValueError(f"Embeddings not found for sequences: {missing_seqs}")

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        emb_a = self.emb_dict.get(seq_a).reshape(-1, self.input_dim)
        emb_b = self.emb_dict.get(seq_b).reshape(-1, self.input_dim)
        
        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        # Prepare the label
        if self.task_type in ['multilabel', 'regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return emb_a, emb_b, label


class EmbedsLabelsDatasetFromDisk(TorchDataset):
    def __init__(
            self,
            hf_dataset,
            col_name='seqs',
            label_col='labels',
            full=False,
            db_path='embeddings.db',
            batch_size=64,
            read_scaler=1000,
            input_dim=768,
            task_type='singlelabel',
            **kwargs
        ): 
        self.seqs, self.labels = hf_dataset[col_name], hf_dataset[label_col]
        self.length = len(self.labels)
        self.max_length = len(max(self.seqs, key=len))
        print('Max length: ', self.max_length)

        self.db_file = db_path
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.full = full

        self.task_type = task_type
        self.read_amt = read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

        self.reset_epoch()

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        cond = False
        for seq in self.seqs:
            if seq not in all_seqs:
                cond = True
            if cond:
                break
        if cond:
            print('Sequences not found in embeddings')
        else:
            print('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs, self.labels))
        random.shuffle(data)
        self.seqs, self.labels = zip(*data)
        self.seqs, self.labels = list(self.seqs), list(self.labels)
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (self.seqs[i],))
            row = result.fetchone()
            emb_data = row[0]
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.input_dim))
            if self.full:
                padding_needed = self.max_length - emb.size(0)
                emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return emb.squeeze(0), label


class EmbedsLabelsDataset(TorchDataset):
    def __init__(self, hf_dataset, emb_dict, col_name='seqs', label_col='labels', task_type='singlelabel', full=False, **kwargs):
        self.embeddings = self.get_embs(emb_dict, hf_dataset[col_name])
        self.full = full
        self.labels = hf_dataset[label_col]
        self.task_type = task_type
        self.max_length = len(max(hf_dataset[col_name], key=len))
        print('Max length: ', self.max_length)

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict[seq]
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = self.embeddings[idx].float()
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = F.pad(emb, (0, 0, 0, padding_needed), value=0)
        return emb.squeeze(0), label
    

class StringLabelDatasetFromHF(TorchDataset):    
    def __init__(self, hf_dataset, col_name='seqs', label_col='labels', **kwargs):
        self.seqs = hf_dataset[col_name]
        self.labels = hf_dataset[label_col]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label
    

class PairStringLabelDatasetFromHF(TorchDataset):
    def __init__(self, hf_dataset, col_a='SeqA', col_b='SeqB', label_col='labels', train=True, **kwargs):
        self.seqs_a, self.seqs_b = hf_dataset[col_a], hf_dataset[col_b]
        self.labels = hf_dataset[label_col]
        self.train = train

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if self.train and random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]


class OneHotCollator:
    def __init__(self, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        # Add X for unknown amino acids, and special CLS and EOS tokens
        alphabet = alphabet + "X"
        alphabet = list(alphabet)
        alphabet.append('cls')
        alphabet.append('eos')
        self.mapping = {token: idx for idx, token in enumerate(alphabet)}
        
    def __call__(self, batch):
        seqs = [ex[0] for ex in batch]
        labels = torch.stack([torch.tensor(ex[1]) for ex in batch])
        
        # Find the longest sequence in the batch (plus 2 for CLS and EOS)
        max_len = max(len(seq) for seq in seqs) + 2
        
        # One-hot encode and pad each sequence
        batch_size = len(seqs)
        one_hot_tensors = []
        attention_masks = []
        
        for seq in seqs:
            seq = ['cls'] + list(seq) + ['eos']
            # Create one-hot encoding for each sequence (including CLS and EOS)
            seq_len = len(seq)
            one_hot = torch.zeros(seq_len, len(self.alphabet))
            
            # Add sequence tokens in the middle
            for pos, token in enumerate(seq):
                if token in self.mapping:
                    one_hot[pos, self.mapping[token]] = 1.0
                else:
                    # For non-canonical amino acids, use the X token
                    one_hot[pos, self.mapping["X"]] = 1.0
            
            # Create attention mask (1 for actual tokens, 0 for padding)
            attention_mask = torch.ones(seq_len)
            
            # Pad to the max length in this batch
            padding_size = max_len - seq_len
            if padding_size > 0:
                padding = torch.zeros(padding_size, len(self.alphabet))
                one_hot = torch.cat([one_hot, padding], dim=0)
                # Add zeros to attention mask for padding
                mask_padding = torch.zeros(padding_size)
                attention_mask = torch.cat([attention_mask, mask_padding], dim=0)
            
            one_hot_tensors.append(one_hot)
            attention_masks.append(attention_mask)
        
        # Stack all tensors in the batch
        embeddings = torch.stack(one_hot_tensors)
        attention_masks = torch.stack(attention_masks)
        
        return {
            'embeddings': embeddings,
            'attention_mask': attention_masks,
            'labels': labels,
        }
