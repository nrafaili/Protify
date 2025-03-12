import torch
import numpy as np
import os
from typing import List
from datasets import Dataset


AMINO_ACIDS = set('LAGVSERTIPDKQNFYMHWCXBUOZ* ')
CODONS = set('aA@bB#$%rRnNdDcCeEqQ^G&ghHiIj+MmlJLkK(fFpPoO=szZwSXTtxWyYuvUV]}) ')
DNA = set('ATCG ')
RNA = set('AUCG ')


def _not_regression(labels): # not a great assumption but works most of the time
    return all(isinstance(label, (int, float)) and label == int(label) for label in labels)


def _encode_labels(labels, tag2id):
    return [torch.tensor([tag2id[tag] for tag in doc], dtype=torch.long) for doc in labels]


def _label_type_checker(labels):
    ex = labels[0]
    if _not_regression(labels):
        if isinstance(ex, list):
            label_type = 'multilabel'
        elif isinstance(ex, int) or isinstance(ex, float):
            label_type = 'singlelabel' # binary or multiclass
    elif isinstance(ex, str):
        label_type = 'string'
    else:
        label_type = 'regression'
    return label_type


def _select_from_sql(c, seq, full, cast_to_torch=True):
    c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (seq,))
    embedding = np.frombuffer(c.fetchone()[0], dtype=np.float32).reshape(1, -1)
    if full:
        embedding = embedding.reshape(len(seq), -1)

    if cast_to_torch:
        embedding = torch.tensor(embedding)
    return embedding


def _select_from_pth(emb_dict, seq, full, cast_to_np=False):
    embedding = emb_dict[seq].reshape(1, -1)
    if full:
        embedding = embedding.reshape(len(seq), -1)

    if cast_to_np:
        embedding = embedding.numpy()
    return embedding


def process_datasets(hf_datasets: List[Dataset], data_names: List[str], max_length: int, trim: bool = False):
    datasets, all_seqs = {}, set()
    for dataset, data_name in zip(hf_datasets, data_names):
        print(f'Processing {data_name}')
        train_set, valid_set, test_set, ppi = dataset
        if trim: # trim by length if necessary
            original_train_size, original_valid_size, original_test_size = len(train_set), len(valid_set), len(test_set)
            if ppi:
                train_set = train_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                valid_set = valid_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
                test_set = test_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= max_length)
            else:
                train_set = train_set.filter(lambda x: len(x['seqs']) <= max_length)
                valid_set = valid_set.filter(lambda x: len(x['seqs']) <= max_length)
                test_set = test_set.filter(lambda x: len(x['seqs']) <= max_length)
        
            print(f'Trimmed {100 * round((original_train_size-len(train_set)) / original_train_size, 2)}% from train')
            print(f'Trimmed {100 * round((original_valid_size-len(valid_set)) / original_valid_size, 2)}% from valid')
            print(f'Trimmed {100 * round((original_test_size-len(test_set)) / original_test_size, 2)}% from test')

        else: # truncate to max_length
            if ppi:
                train_set = train_set.map(lambda x: {'SeqA': x['SeqA'][:max_length], 'SeqB': x['SeqB'][:max_length]})
                valid_set = valid_set.map(lambda x: {'SeqA': x['SeqA'][:max_length], 'SeqB': x['SeqB'][:max_length]})
                test_set = test_set.map(lambda x: {'SeqA': x['SeqA'][:max_length], 'SeqB': x['SeqB'][:max_length]})
            else:
                train_set = train_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                valid_set = valid_set.map(lambda x: {'seqs': x['seqs'][:max_length]})
                test_set = test_set.map(lambda x: {'seqs': x['seqs'][:max_length]})

        # sanitize
        if ppi:
            train_set = train_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS), 'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
            valid_set = valid_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS), 'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
            test_set = test_set.map(lambda x: {'SeqA': ''.join(aa for aa in x['SeqA'] if aa in AMINO_ACIDS), 'SeqB': ''.join(aa for aa in x['SeqB'] if aa in AMINO_ACIDS)})
            all_seqs.update(train_set['SeqA'] + train_set['SeqB'])
            all_seqs.update(valid_set['SeqA'] + valid_set['SeqB'])
            all_seqs.update(test_set['SeqA'] + test_set['SeqB'])
        else:
            train_set = train_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
            valid_set = valid_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
            test_set = test_set.map(lambda x: {'seqs': ''.join(aa for aa in x['seqs'] if aa in AMINO_ACIDS)})
            all_seqs.update(train_set['seqs'])
            all_seqs.update(valid_set['seqs'])
            all_seqs.update(test_set['seqs'])
            
        # confirm the type of labels
        check_labels = valid_set['labels']
        label_type = _label_type_checker(check_labels)

        if label_type == 'string': # might be string or multilabel
            example = valid_set['labels'][0]
            try:
                import ast
                new_ex = ast.literal_eval(example)
                if isinstance(new_ex, list): # if ast runs correctly and is now a list it is multilabel labels
                    label_type = 'multilabel'
                    train_set = train_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                    valid_set = valid_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
                    test_set = test_set.map(lambda ex: {'labels': ast.literal_eval(ex['labels'])})
            except:
                label_type = 'string' # if ast throws error it is actually string

        if label_type == 'string': # if still string, it's for tokenwise classification
            train_labels = train_set['labels']
            unique_tags = set(tag for doc in train_labels for tag in doc)
            tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
            train_set = train_set.map(lambda ex: {'labels': _encode_labels(ex['labels'], tag2id=tag2id)})
            valid_set = valid_set.map(lambda ex: {'labels': _encode_labels(ex['labels'], tag2id=tag2id)})
            test_set = test_set.map(lambda ex: {'labels': _encode_labels(ex['labels'], tag2id=tag2id)})
            label_type = 'tokenwise'
            num_labels = len(unique_tags)
        else:
            if label_type == 'regression':
                num_labels = 1
            else: # if classification, get the total number of leabels
                try:
                    num_labels = len(train_set['labels'][0])
                except:
                    unique = np.unique(train_set['labels'])
                    max_label = max(unique) # sometimes there are missing labels
                    full_list = np.arange(0, max_label+1)
                    num_labels = len(full_list)
        datasets[data_name] = (train_set, valid_set, test_set, num_labels, label_type, ppi)

    all_seqs = list(all_seqs)
    all_seqs = sorted(all_seqs, key=len, reverse=True) # longest first
    return datasets, all_seqs


def get_embedding_dim_sql(save_path, full, test_seq, max_length):
    import sqlite3
    if len(test_seq) > max_length - 2:
        test_seq_len = max_length
    else:
        test_seq_len = len(test_seq) + 2
    
    with sqlite3.connect(save_path) as conn:
        c = conn.cursor()
        c.execute("SELECT embedding FROM embeddings WHERE sequence = ?", (test_seq,))
        test_embedding = c.fetchone()[0]
        test_embedding = torch.tensor(np.frombuffer(test_embedding, dtype=np.float32).reshape(1, -1))
    if full:
        test_embedding = test_embedding.reshape(test_seq_len, -1)
    embedding_dim = test_embedding.shape[-1]
    return embedding_dim


def get_embedding_dim_pth(emb_dict, full, test_seq, max_length):
    if len(test_seq) > max_length - 2:
        test_seq_len = max_length
    else:
        test_seq_len = len(test_seq) + 2

    test_embedding = emb_dict[test_seq].reshape(1, -1)
    if full:
        test_embedding = test_embedding.reshape(test_seq_len, -1)
    embedding_dim = test_embedding.shape[-1]
    return embedding_dim


def build_vector_numpy_dataset_from_embeddings(
        train_seqs,
        valid_seqs,
        test_seqs,
        sql,
        save_dir,
        model_name,
        full,
    ):
    train_array, valid_array, test_array = [], [], []
    if sql:
        import sqlite3
        save_path = os.path.join(save_dir, f'{model_name}_{full}.db')
        with sqlite3.connect(save_path) as conn:
            c = conn.cursor()
            for seq in train_seqs:
                embedding = _select_from_sql(c, seq, full, cast_to_torch=False)
                train_array.append(embedding)

            for seq in valid_seqs:
                embedding = _select_from_sql(c, seq, full, cast_to_torch=False)
                valid_array.append(embedding)

            for seq in test_seqs:
                embedding = _select_from_sql(c, seq, full, cast_to_torch=False)
                test_array.append(embedding)
    else:
        save_path = os.path.join(save_dir, f'{model_name}_{full}.pth')
        emb_dict = torch.load(save_path)
        for seq in train_seqs:
            embedding = _select_from_pth(emb_dict, seq, full, cast_to_np=True)
            train_array.append(embedding)
            
        for seq in valid_seqs:
            embedding = _select_from_pth(emb_dict, seq, full, cast_to_np=True)
            valid_array.append(embedding)

        for seq in test_seqs:
            embedding = _select_from_pth(emb_dict, seq, full, cast_to_np=True)
            test_array.append(embedding)
        del emb_dict

    train_array = np.concatenate(train_array, axis=0)
    valid_array = np.concatenate(valid_array, axis=0)
    test_array = np.concatenate(test_array, axis=0)
    
    if full: # average over the length of the sequence
        train_array = np.mean(train_array, axis=1)
        valid_array = np.mean(valid_array, axis=1)
        test_array = np.mean(test_array, axis=1)

    print('Numpy dataset shapes')
    print(train_array.shape, valid_array.shape, test_array.shape)
    return train_array, valid_array, test_array


def build_pair_vector_numpy_dataset_from_embeddings(
        train_seqs_a,
        train_seqs_b,
        valid_seqs_a,
        valid_seqs_b,
        test_seqs_a,
        test_seqs_b,
        sql,
        save_dir,
        model_name,
        full,
    ):
    import random

    def _random_order(seq_a, seq_b):
        if random.random() < 0.5:
            return seq_a, seq_b
        else:
            return seq_b, seq_a

    train_array, valid_array, test_array = [], [], []
    if sql:
        import sqlite3
        save_path = os.path.join(save_dir, f'{model_name}_{full}.db')
        with sqlite3.connect(save_path) as conn:
            c = conn.cursor()
            for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
                seq_a, seq_b = _random_order(seq_a, seq_b)
                embedding_a = _select_from_sql(c, seq_a, full, cast_to_torch=False)
                embedding_b = _select_from_sql(c, seq_b, full, cast_to_torch=False)
                train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
                seq_a, seq_b = _random_order(seq_a, seq_b)
                embedding_a = _select_from_sql(c, seq_a, full, cast_to_torch=False)
                embedding_b = _select_from_sql(c, seq_b, full, cast_to_torch=False)
                valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

            for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
                seq_a, seq_b = _random_order(seq_a, seq_b)
                embedding_a = _select_from_sql(c, seq_a, full, cast_to_torch=False)
                embedding_b = _select_from_sql(c, seq_b, full, cast_to_torch=False)
                test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
    else:
        save_path = os.path.join(save_dir, f'{model_name}_{full}.pth')
        emb_dict = torch.load(save_path)
        for seq_a, seq_b in zip(train_seqs_a, train_seqs_b):
            seq_a, seq_b = _random_order(seq_a, seq_b)
            embedding_a = _select_from_pth(emb_dict, seq_a, full, cast_to_np=True)
            embedding_b = _select_from_pth(emb_dict, seq_b, full, cast_to_np=True)
            train_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

        for seq_a, seq_b in zip(valid_seqs_a, valid_seqs_b):
            seq_a, seq_b = _random_order(seq_a, seq_b)
            embedding_a = _select_from_pth(emb_dict, seq_a, full, cast_to_np=True)
            embedding_b = _select_from_pth(emb_dict, seq_b, full, cast_to_np=True)
            valid_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))

        for seq_a, seq_b in zip(test_seqs_a, test_seqs_b):
            embedding_a = _select_from_pth(emb_dict, seq_a, full, cast_to_np=True)
            embedding_b = _select_from_pth(emb_dict, seq_b, full, cast_to_np=True)
            test_array.append(np.concatenate([embedding_a, embedding_b], axis=-1))
        del emb_dict

    train_array = np.concatenate(train_array, axis=0)
    valid_array = np.concatenate(valid_array, axis=0)
    test_array = np.concatenate(test_array, axis=0)
    
    if full: # average over the length of the sequence
        train_array = np.mean(train_array, axis=1)
        valid_array = np.mean(valid_array, axis=1)
        test_array = np.mean(test_array, axis=1)

    print('Numpy dataset shapes')
    print(train_array.shape, valid_array.shape, test_array.shape)
    return train_array, valid_array, test_array


def labels_to_numpy(labels):
    if isinstance(labels[0], list):
        return np.array(labels)
    else:
        return np.array([labels])


def prepare_scikit_dataset(dataset, sql):
    train_set, valid_set, test_set, num_labels, label_type, ppi = dataset

    if ppi:
        X_train, X_valid, X_test = build_pair_vector_numpy_dataset_from_embeddings(
            train_set['SeqA'],
            train_set['SeqB'],
            valid_set['SeqA'],
            valid_set['SeqB'],
            test_set['SeqA'],
            test_set['SeqB'],
        )

