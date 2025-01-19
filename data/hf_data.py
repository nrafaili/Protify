from typing import List
from datasets import load_dataset
from dataclasses import dataclass
from .data_utils import process_datasets
from .supported_datasets import supported_datasets, possible_with_vector_reps


@dataclass
class HFDataArguments:
    """
    Args:
    data_paths: List[str]
        paths to the datasets
    max_length: int
        max length of sequences
    trim: bool
        whether to trim sequences to max_length
    """
    def __init__(
            self,
            data_paths: List[str],
            max_length: int = 1024,
            trim: bool = False,
            **kwargs):
        self.data_paths = []
        if data_paths[0] == 'vector_benchmark':
            self.data_paths = possible_with_vector_reps

        for data_path in data_paths:
            if data_path in supported_datasets:
                self.data_paths.append(supported_datasets[data_path])
            else:
                self.data_paths.append(data_path)
        self.max_length = max_length
        self.trim = trim


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
        print(f'Loading {data_name}')
        dataset = load_dataset(data_path)
        train_set, valid_set, test_set = dataset['train'], dataset['valid'], dataset['test']
        datasets.append((train_set, valid_set, test_set, ppi))
    return process_datasets(hf_datasets=datasets, data_name=data_name, max_length=args.max_length, trim=args.trim)
