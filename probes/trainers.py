import torch
import os
from torchinfo import summary
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from data.torch_classes import (
    embeds_labels_collator_builder,
    pair_embeds_labels_collator_builder,
    EmbedsLabelsDatasetFromDisk,
    PairEmbedsLabelsDatasetFromDisk,
    EmbedsLabelsDataset,
    PairEmbedsLabelsDataset
)
from probes.get_probe import get_probe


@dataclass
class TrainerArguments:
    def __init__(
            self,
            model_save_dir: str,
            num_epochs: int = 200,
            batch_size: int = 64,
            gradient_accumulation_steps: int = 1,
            lr: float = 1e-4,
            weight_decay: float = 0.00,
            task_type: str = 'regression',
            patience: int = 3,
            read_scaler: int = 1000,
            save: bool = False,
            **kwargs
    ):
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.patience = patience
        self.save = save
        self.read_scaler = read_scaler

    def __call__(self):
        if '/' in self.model_save_dir:
            save_dir = self.model_save_dir.split('/')[-1]
        else:
            save_dir = self.model_save_dir
        return TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=3,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_steps=1000,
            metric_for_best_model='spearman_rho' if self.task_type == 'regression' else 'mcc',
            greater_is_better=True,
            load_best_model_at_end=True,
        )


def train_probe(
        trainer_args,
        embedding_args,
        probe_args,
        tokenizer,
        train_dataset,
        valid_dataset,
        test_dataset,
        model_name,
        emb_dict=None,
        ppi=False,
    ):
    probe = get_probe(probe_args)
    summary(probe)
    full = embedding_args.matrix_embed
    db_path = os.path.join(embedding_args.embedding_save_dir, f'{model_name}_{full}.db')
    print(embedding_args.sql, ppi, full)

    if embedding_args.sql:
        if ppi:
            if full:
                raise ValueError('Full matrix embeddings not currently supported for SQL and PPI') # TODO: Implement
            DatasetClass = PairEmbedsLabelsDatasetFromDisk
            collate_builder = pair_embeds_labels_collator_builder
        else:
            DatasetClass = EmbedsLabelsDatasetFromDisk
            collate_builder = embeds_labels_collator_builder
    else:
        if ppi:
            DatasetClass = PairEmbedsLabelsDataset
            collate_builder = pair_embeds_labels_collator_builder
        else:
            DatasetClass = EmbedsLabelsDataset
            collate_builder = embeds_labels_collator_builder

    """
    For collator need to pass tokenizer, full, task_type
    For dataset need to pass hf_dataset, col_a, col_b, label_col, input_dim, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
    """
    data_collator = collate_builder(tokenizer=tokenizer, full=full, task_type=probe_args.task_type)
    train_dataset = DatasetClass(
        hf_dataset=train_dataset,
        input_dim=probe_args.input_dim,
        task_type=probe_args.task_type,
        db_path=db_path,
        emb_dict=emb_dict,
        batch_size=trainer_args.batch_size,
        read_scaler=trainer_args.read_scaler,
        full=full,
        train=True
    )
    hf_trainer_args = trainer_args()
    trainer = Trainer(
        model=probe,
        args=hf_trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=trainer_args.patience)]
    )
    ### TODO logging
    metrics = trainer.evaluate(test_dataset)
    print(f'Initial metrics: \n{metrics}\n')

    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    print(f'Final metrics: \n{metrics}\n')

    if trainer_args.save:
        try:
            trainer.model.push_to_hub(trainer_args.model_save_dir, private=True)
        except Exception as e:
            print(f'Error saving model: {e}')

    trainer.accelerate.free_memory()
    torch.cuda.empty_cache()

