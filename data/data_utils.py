import torch
import numpy as np
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


def process_datasets(hf_datasets: List[Dataset], data_name: str, max_length: int, trim: bool = False):
    datasets, all_seqs = {}, set()
    for dataset in hf_datasets:
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

    all_seqs = list(all_seqs)
    all_seqs = sorted(all_seqs, key=len, reverse=True) # longest first
    datasets[data_name] = (train_set, valid_set, test_set, num_labels, label_type, ppi)
    return datasets, all_seqs
