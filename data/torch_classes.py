### imports
import random
import torch
import numpy as np
import sqlite3
from tqdm.auto import tqdm
from torch.utils.data import Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence


def pair_collator(inputs, tokenizer):
    seqa = [f[0] for f in inputs]
    seqb = [f[1] for f in inputs]
    labels = [f[2] for f in inputs]
    a = tokenizer(seqa, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=True)
    b = tokenizer(seqb, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=True)
    max_batch_length = len(max(labels, key=len))
    labels = torch.stack([torch.tensor(label + [-100] * (max_batch_length - len(label))) for label in labels])
    return {
        'seq_a': a,
        'seq_b': b,
        'labels': labels
    }


def ppi_from_embs_collate_wrapper(full_attention):
    def collate_fn(batch):
        emb_a_list = [item['emb_a'] for item in batch]
        emb_b_list = [item['emb_b'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        if full_attention:
            # Concatenate emb_a and emb_b before padding
            concatenated_embs = [torch.cat([emb_a, emb_b], dim=0) for emb_a, emb_b in zip(emb_a_list, emb_b_list)]
            # Pad the concatenated embeddings
            embeddings = pad_sequence(concatenated_embs, batch_first=True, padding_value=0)
            return {'hidden_state': embeddings, 'labels': labels}
        else:
            # Pad emb_a and emb_b separately
            emb_a_padded = pad_sequence(emb_a_list, batch_first=True, padding_value=0)
            emb_b_padded = pad_sequence(emb_b_list, batch_first=True, padding_value=0)
            # Pad sequences to the same length
            max_len = max(emb_a_padded.size(1), emb_b_padded.size(1))
            emb_a_padded = torch.nn.functional.pad(emb_a_padded, (0, 0, 0, max_len - emb_a_padded.size(1)), value=0)
            emb_b_padded = torch.nn.functional.pad(emb_b_padded, (0, 0, 0, max_len - emb_b_padded.size(1)), value=0)
            # Concatenate along the feature dimension
            embeddings = torch.cat([emb_a_padded, emb_b_padded], dim=2)
            return {'hidden_state_a': emb_a_padded, 'hidden_state_b': emb_b_padded, 'labels': labels}
    return collate_fn


### Standard
def collate_seq_labels(tokenizer):
    def _collate_fn(batch):
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]
        batch = tokenizer(seqs,
                          padding='longest',
                          truncation=False,
                          return_tensors='pt',
                          add_special_tokens=True)
        batch['labels'] = torch.stack([torch.tensor(label, dtype=torch.float) for label in labels])
        return batch
    return _collate_fn


def collate_fn_embeds(full=False, max_length=512, task_type='tokenwise'):
    def _collate_fn(batch):
        ppi = len(batch[0]) == 3
        if ppi:
            embeds_a = torch.stack([ex[0] for ex in batch])
            embeds_b = torch.stack([ex[1] for ex in batch])
            labels = torch.stack([ex[2] for ex in batch])
            embeds = torch.cat([embeds_a, embeds_b], dim=-1)
        else:
            embeds = torch.stack([ex[0] for ex in batch])
            labels = torch.stack([ex[1] for ex in batch])
    
        if full and task_type == 'tokenwise':
            padded_labels = []
            for label in labels:
                padding_size = max_length - label.size(0)
                if padding_size > 0:
                    padding = torch.full((padding_size,), -100, dtype=label.dtype)
                    padded_label = torch.cat((label.squeeze(-1), padding))
                else:
                    padded_label = label[:max_length].squeeze(-1)  # Truncate if longer than max_length
                padded_labels.append(padded_label)
            labels = torch.stack(padded_labels)
        return {
            'embeddings': embeds,
            'labels': labels
        }
    return _collate_fn



class PPIDatasetEmbedsFromDisk(TorchDataset):
    def __init__(self, args, seqs_a, seqs_b, labels, input_dim=768, task_type='regression', all_seqs=None):
        self.db_file = args.db_path
        self.batch_size = args.batch_size
        self.emb_dim = input_dim
        self.full = args.full
        self.seqs_a, self.seqs_b, self.labels = seqs_a, seqs_b, labels
        self.length = len(labels)
        self.read_amt = args.read_scaler * self.batch_size
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0
        self.task_type = task_type

        if all_seqs:
            print('Pre shuffle check')
            self.check_seqs(all_seqs)
        self.reset_epoch()
        if all_seqs:
            print('Post shuffle check')
            self.check_seqs(all_seqs)

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
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.emb_dim))
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


class PPIDatasetEmbeds(TorchDataset):
    def __init__(self, args, emb_dict, seqs_a, seqs_b, labels, input_dim, task_type='regression'):
        self.emb_dim = input_dim
        self.task_type = task_type
        self.full = args.full
        # Combine seqs_a and seqs_b to find all unique sequences needed
        needed_seqs = set(seqs_a + seqs_b)
        # Filter emb_dict to keep only the necessary embeddings
        self.emb_dict = {seq: emb_dict[seq] for seq in needed_seqs if seq in emb_dict}
        # Check for any missing embeddings
        missing_seqs = needed_seqs - self.emb_dict.keys()
        if missing_seqs:
            raise ValueError(f"Embeddings not found for sequences: {missing_seqs}")
        self.seqs_a = seqs_a
        self.seqs_b = seqs_b
        self.labels = labels

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        seq_a = self.seqs_a[idx]
        seq_b = self.seqs_b[idx]
        emb_a = torch.tensor(self.emb_dict.get(seq_a).reshape(-1, self.emb_dim))
        emb_b = torch.tensor(self.emb_dict.get(seq_b).reshape(-1, self.emb_dim))
        
        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        # Prepare the label
        if self.task_type in ['multilabel', 'regression']:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return emb_a, emb_b, label

    
### SQL
class FineTuneDatasetEmbedsFromDisk(TorchDataset):
    def __init__(self, args, seqs, labels, input_dim=768, task_type='binary', all_seqs=None): 
        self.db_file = args.db_path
        self.batch_size = args.batch_size
        self.emb_dim = input_dim
        self.full = args.full
        self.seqs, self.labels = seqs, labels
        self.length = len(labels)
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.task_type = task_type
        self.read_amt = args.read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

        if all_seqs:
            print('Pre shuffle check')
            self.check_seqs(all_seqs)
        self.reset_epoch()
        if all_seqs:
            print('Post shuffle check')
            self.check_seqs(all_seqs)

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
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.emb_dim))
            if self.full:
                padding_needed = self.max_length - emb.size(0)
                emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
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


class FineTuneDatasetEmbeds(TorchDataset):
    def __init__(self, args, seqs, labels, emb_dict, task_type='binary'):
        self.embeddings = self.get_embs(emb_dict, seqs)
        self.labels = labels
        self.task_type = task_type
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.full = args.full

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict.get(seq)
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float)
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
        return emb.squeeze(0), label
    

class SequenceLabelDatasetFromHF(TorchDataset):    
    def __init__(self, dataset, col_name='seqs', label_col='labels'):
        self.seqs = dataset[col_name]
        self.labels = dataset[label_col]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label
    

class PairDatasetTrainHF(TorchDataset):
    def __init__(self, data, col_a, col_b, label_col):
        self.seqs_a = data[col_a]
        self.seqs_b = data[col_b]
        self.labels = data[label_col]

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        seq_a, seq_b = self.seqs_a[idx], self.seqs_b[idx]
        if random.random() < 0.5:
            seq_a, seq_b = seq_b, seq_a
        return seq_a, seq_b, self.labels[idx]
