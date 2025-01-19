import logging
import functools
import json
import csv
import os
import datetime
import ast
import random
import string
from pathlib import Path
from types import SimpleNamespace


def log_method_calls(func):
    """Decorator to log each call of the decorated method."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.logger.info(f"Called method: {func.__name__}")
        return func(self, *args, **kwargs)
    return wrapper


class MetricsLogger:
    """
    Logs method calls to a text file, and keeps a TSV-based matrix of metrics:
      - Rows = dataset names
      - Columns = model names
      - Cells = JSON-encoded dictionaries of metrics
    """

    def __init__(self, args):
        self.logger_args = args
        self._section_break = '\n' + '=' * 55 + '\n'

    def _start_file(self):
        args = self.logger_args
        self.log_dir = args.log_dir
        self.results_dir = args.results_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Generate random ID with date and 4-letter code
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=4))
        date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.random_id = f"{random_letters}_{date_str}"
        
        if args.replay_path is not None:
            self.random_id = 'replay_' + args.replay_path.split('/')[-1].split('.')[0]
        self.log_file = os.path.join(self.log_dir, f"{self.random_id}.txt")
        self.results_file = os.path.join(self.results_dir, f"{self.random_id}.tsv")

    def _minimial_logger(self):
        # Set up a minimal logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, mode='a')
            handler.setLevel(logging.INFO)
            # Simple formatter without duplicating date/time
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # TSV tracking
        self.results_file = self.results_file
        self.logger_data_tracking = {}  # { dataset_name: { model_name: metrics_dict } }

    def _write_args(self):
        with open(self.log_file, 'a') as f:
            f.write(self._section_break)
            for k, v in self.logger_args.__dict__.items():
                if 'token' not in k.lower() and 'api' not in k.lower():
                    f.write(f"{k}:\t{v}\n")
            f.write(self._section_break)

    def start_log_main(self):
        self._start_file()

        with open(self.log_file, 'w') as f:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.logger_args.replay_path is not None:
                message = f'=== REPLAY OF {self.logger_args.replay_path} ===\n'
                f.write(message)
            header = f"=== Logging session started at {now} ===\n"
            f.write(header)
            self._write_args()

        self._minimial_logger()

    def start_log_gui(self):
        self._start_file()
        with open(self.log_file, 'w') as f:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if self.logger_args.replay_path is not None:
                message = f'=== REPLAY OF {self.logger_args.replay_path} ===\n'
                f.write(message)
            header = f"=== Logging session started at {now} ===\n"
            f.write(header)
            f.write(self._section_break)
        self._minimial_logger()

    def load_tsv(self):
        """Load existing TSV data into self.logger_data_tracking (row=dataset, col=model)."""
        with open(self.results_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if not header:
                return

            model_names = header[1:]
            for row in reader:
                if row:
                    ds = row[0]
                    self.logger_data_tracking[ds] = {}
                    for i, model in enumerate(model_names, start=1):
                        cell_val = row[i].strip()
                        if cell_val:
                            try:
                                self.logger_data_tracking[ds][model] = json.loads(cell_val)
                            except json.JSONDecodeError:
                                self.logger_data_tracking[ds][model] = {"_raw": cell_val}

    def write_results(self):
        datasets = sorted(self.logger_data_tracking.keys())
        
        # Calculate average eval_loss for each model
        model_scores = {}
        for model in {m for ds in self.logger_data_tracking.values() for m in ds.keys()}:
            losses = []
            for ds in datasets:
                if model in self.logger_data_tracking[ds] and 'eval_loss' in self.logger_data_tracking[ds][model]:
                    losses.append(self.logger_data_tracking[ds][model]['eval_loss'])
            if losses:
                model_scores[model] = sum(losses) / len(losses)
            else:
                model_scores[model] = float('inf')  # Models without eval_loss go last
        
        # Sort models by average eval_loss
        model_names = sorted(model_scores.keys(), key=lambda m: model_scores[m])

        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["dataset"] + model_names)
            for ds in datasets:
                row = [ds]
                for model in model_names:
                    metrics = self.logger_data_tracking[ds].get(model, {})
                    row.append(json.dumps(metrics))
                writer.writerow(row)

    def log_metrics(self, dataset, model, metrics_dict):
        # remove keys with time or second in them
        metrics_dict = {k: v for k, v in metrics_dict.items() if 'time' not in k.lower() and 'second' not in k.lower()}
        self.logger.info(f"Storing metrics for {dataset}/{model}: {metrics_dict}")
        self.logger_data_tracking.setdefault(dataset, {})[model] = metrics_dict


class LogReplayer:
    def __init__(self, log_file_path):
        self.log_file = Path(log_file_path)
        self.arguments = {}
        self.method_calls = []

    def parse_log(self):
        """
        Reads the log file line by line. Extracts:
          1) Global arguments into self.arguments
          2) Method calls into self.method_calls (in order)
        """
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        with open(self.log_file, 'r') as file:
            header = next(file)
            for line in file:
                if line.startswith('='):
                    continue
                elif line.startswith('INFO'):
                    method = line.split(': ')[-1].strip()
                    self.method_calls.append(method)
                elif ':\t' in line:
                    key, value = line.split(':\t')
                    key, value = key.strip(), value.strip()
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
                    self.arguments[key] = value

        return SimpleNamespace(**self.arguments)

    def run_replay(self, target_obj):
        """
        Replays the collected method calls on `target_obj`.
        `target_obj` is an instance of the class/script that we want to replay.
        """
        for method in self.method_calls:
            print(f"Replaying call to: {method}()")
            func = getattr(target_obj, method, None)
            if not func:
                print(f"Warning: {method} not found on target object.")
                continue
            func()
