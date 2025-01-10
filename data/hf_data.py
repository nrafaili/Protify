from typing import List
from datasets import load_dataset
from dataclasses import dataclass
from .data_utils import process_datasets


@dataclass
class HFDataArguments:
    """
    Args:
    data_paths: List[str]
        paths to the datasets
    max_len: int
        max length of sequences
    trim: bool
        whether to trim sequences to max_len
    """
    data_paths: List[str]
    max_len: int = 1024
    trim: bool = False


def get_hf_data(args: HFDataArguments):
    """
    Currently only handles datasets with columns 'seqs' and 'labels'
    OR
    'SeqA', 'SeqB', and 'labels' for PPI datasets
    """
    datasets = []
    for data_path in args.data_paths:
        data_name = data_path.split('/')[-1]
        ppi = 'ppi' in data_name.lower()
        dataset = load_dataset(data_path)
        train_set, valid_set, test_set = dataset['train'], dataset['valid'], dataset['test']
        datasets.append((train_set, valid_set, test_set, ppi))
    return process_datasets(hf_datasets=datasets, data_name=data_name, max_len=args.max_len, trim=args.trim)
