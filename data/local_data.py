import os
from glob import glob
from pandas import read_csv, read_excel
from datasets import Dataset
from typing import List
from dataclasses import dataclass
from .data_utils import process_datasets


@dataclass
class LocalDataArguments:
    """
    Args:
    data_dirs: List[str]
        directory with train, valid, and test datasets of the same format named train, valid, test (.csv, .tsv, etc.)
    delimiter: str
        delimiter for the dataset
    col_names: List[str]
        column names for the dataset
    max_len: int
        max length of sequences
    trim: bool
        whether to trim sequences to max_len
    """
    def __init__(
            self,
            data_dirs: List[str],
            delimiter: str = ',',
            col_names: List[str] = ['seqs', 'labels'],
            max_len: int = 1024,
            trim: bool = False,
            **kwargs):
        self.data_dirs = data_dirs
        self.delimiter = delimiter
        self.col_names = col_names
        self.max_len = max_len
        self.trim = trim


def get_local_data(args: LocalDataArguments):
    """
    Supports .csv, .tsv, .txt
    TODO fasta, fa, fna, etc.
    """
    datasets = []
    for data_dir in args.data_dirs:
        data_name = data_dir.split('/')[-2]
        ppi = 'ppi' in data_dir.lower()
        train_path = glob(os.path.join(data_dir, 'train.*'))[0]
        valid_path = glob(os.path.join(data_dir, 'valid.*'))[0]
        test_path = glob(os.path.join(data_dir, 'test.*'))[0]
        if '.xlsx' in train_path:
            train_set = read_excel(train_path)
            valid_set = read_excel(valid_path)
            test_set = read_excel(test_path)
        else:
            train_set = read_csv(train_path, delimiter=args.delimiter, names=args.col_names)
            valid_set = read_csv(valid_path, delimiter=args.delimiter, names=args.col_names)
            test_set = read_csv(test_path, delimiter=args.delimiter, names=args.col_names)

        train_set = Dataset.from_pandas(train_set)
        valid_set = Dataset.from_pandas(valid_set)
        test_set = Dataset.from_pandas(test_set)
        datasets.append((train_set, valid_set, test_set))
    return process_datasets(datasets=datasets, data_name=data_name, max_len=args.max_len, trim=args.trim, ppi=ppi)
