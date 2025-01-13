from transformers import TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from ..data.torch_classes import (
    
)


@dataclass
class TrainerArguments:
    output_dir: str
    num_epochs: int = 200
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    lr: float = 1e-4
    task_type: str = 'regression'    
    patience: int = 3

    def __call__(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            weight_decay=0.01,
            save_total_limit=3,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_steps=100,
            metric_for_best_model='spearman_rho' if self.task_type == 'regression' else 'mcc',
            greater_is_better=True,
            load_best_model_at_end=True,
        )


def get_trainer(embedding_args, model, train_dataset, valid_dataset, test_dataset):
    if embedding_args.sql:
        