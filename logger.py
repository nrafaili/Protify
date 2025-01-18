import logging
import functools
import json
import csv
import os
import datetime


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

    def __init__(self, main_log='main_log.txt', tsv_file='metrics.tsv'):
        # Write the date/time once if we're creating a fresh log
        if not os.path.exists(main_log):
            with open(main_log, 'w') as f:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"=== Logging session started at {now} ===\n")

        # Set up a minimal logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(main_log, mode='a')
            handler.setLevel(logging.INFO)
            # Simple formatter without duplicating date/time
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # TSV tracking
        self.tsv_file = tsv_file
        self.data = {}  # { dataset_name: { model_name: metrics_dict } }

        # If TSV exists, load it
        if os.path.exists(tsv_file):
            self.load_tsv()

    def load_tsv(self):
        """Load existing TSV data into self.data (row=dataset, col=model)."""
        with open(self.tsv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            if not header:
                return

            model_names = header[1:]
            for row in reader:
                if row:
                    ds = row[0]
                    self.data[ds] = {}
                    for i, model in enumerate(model_names, start=1):
                        cell_val = row[i].strip()
                        if cell_val:
                            try:
                                self.data[ds][model] = json.loads(cell_val)
                            except json.JSONDecodeError:
                                self.data[ds][model] = {"_raw": cell_val}

    def write_tsv(self):
        """Write in-memory data to the TSV file."""
        datasets = sorted(self.data.keys())
        model_names = sorted({m for ds in self.data.values() for m in ds.keys()})

        with open(self.tsv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["dataset"] + model_names)
            for ds in datasets:
                row = [ds]
                for model in model_names:
                    metrics = self.data[ds].get(model, {})
                    row.append(json.dumps(metrics))
                writer.writerow(row)

    def log_metrics(self, dataset, model, metrics_dict):
        """Stores metrics for a (dataset, model) combination and logs it."""
        self.logger.info(f"Storing metrics for {dataset}/{model}: {metrics_dict}")
        self.data.setdefault(dataset, {})[model] = metrics_dict

    @log_method_calls
    def example_method(self):
        """Just a method to show that the call is logged."""
        pass

# -------------------------
# Example Usage (Demo)
# -------------------------
if __name__ == "__main__":
    logger_instance = MetricsLogger()

    # Show method call logging
    logger_instance.example_method()

    # Store some metrics
    logger_instance.log_metrics("Dataset1", "ModelA", {"accuracy": 0.92, "f1": 0.88})
    logger_instance.log_metrics("Dataset1", "ModelB", {"accuracy": 0.87, "f1": 0.85})
    logger_instance.log_metrics("Dataset2", "ModelA", {"accuracy": 0.75, "f1": 0.70})

    # Write everything to TSV
    logger_instance.write_tsv()
    print("Done! Check 'main_log.txt' for logs, and 'metrics.tsv' for metrics.")
