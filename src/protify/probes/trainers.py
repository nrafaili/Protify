import torch
import os
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
from probes.hybrid_probe import HybridProbe, HybridProbeConfig
from data.dataset_classes import (
    EmbedsLabelsDatasetFromDisk,
    PairEmbedsLabelsDatasetFromDisk,
    EmbedsLabelsDataset,
    PairEmbedsLabelsDataset,
    StringLabelDataset,
    PairStringLabelDataset,
)
from data.data_collators import (
    EmbedsLabelsCollator,
    PairEmbedsLabelsCollator,
    PairCollator_input_ids,
    StringLabelsCollator,
)
from visualization.ci_plots import regression_ci_plot, classification_ci_plot
from utils import print_message
from metrics import get_compute_metrics


@dataclass
class TrainerArguments:
    def __init__(
            self,
            model_save_dir: str,
            num_epochs: int = 200,
            trainer_batch_size: int = 64,
            gradient_accumulation_steps: int = 1,
            lr: float = 1e-4,
            weight_decay: float = 0.00,
            task_type: str = 'regression',
            patience: int = 3,
            read_scaler: int = 1000,
            save_model: bool = False,
            seed: int = 42,
            train_data_size: int = 100,
            plots_dir: str = None,
            **kwargs
    ):
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        self.batch_size = trainer_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.task_type = task_type
        self.patience = patience
        self.save = save_model
        self.read_scaler = read_scaler
        self.seed = seed
        self.train_data_size = train_data_size
        self.plots_dir = plots_dir

    def __call__(self):
        if self.train_data_size > 50000:
            eval_strats = {
                'eval_strategy': 'steps',
                'eval_steps': 5000,
                'save_strategy': 'steps',
                'save_steps': 5000,
            }
        else:
            eval_strats = {
                'eval_strategy': 'epoch',
                'save_strategy': 'epoch',
            }

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
            logging_steps=1000,
            report_to='none',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=self.seed,
            **eval_strats
        )


def train(
        model,
        task_type,
        trainer_args,
        data_collator,
        train_dataset,
        valid_dataset,
        test_dataset,
        model_name,
        data_name,
        log_id,
    ):

    compute_metrics = get_compute_metrics(task_type)
    trainer_args.train_data_size = len(train_dataset)
    hf_trainer_args = trainer_args()
    ### TODO add options for optimizers and schedulers
    trainer = Trainer(
        model=model,
        args=hf_trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=trainer_args.patience)]
    )
    metrics = trainer.evaluate(test_dataset)
    print_message(f'Initial metrics: {metrics}')

    trainer.train()

    valid_metrics = trainer.evaluate(valid_dataset)
    print_message(f'Final validation metrics: {valid_metrics}')

    y_pred, y_true, test_metrics = trainer.predict(test_dataset)
    y_pred, y_true = y_pred.astype(np.float32), y_true.astype(np.float32)
    print_message(f'y_pred: {y_pred.shape}\ny_true: {y_true.shape}\nFinal test metrics: \n{test_metrics}\n')

    output_dir = os.path.join(trainer_args.plots_dir, log_id)
    os.makedirs(output_dir, parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}.png")

    if task_type == 'regression':
        regression_ci_plot(y_true, y_pred, save_path)
    else:
        classification_ci_plot(y_true, y_pred, save_path)

    if trainer_args.save:
        try:
            trainer.model.push_to_hub(trainer_args.model_save_dir, private=True)
        except Exception as e:
            print_message(f'Error saving model: {e}')

    model = trainer.model.cpu()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()
    return model, valid_metrics, test_metrics


def train_probe(
        model,
        tokenizer,
        trainer_args,
        embedding_args,
        model_name,
        data_name,
        input_dim,
        task_type,
        train_dataset,
        valid_dataset,
        test_dataset,
        emb_dict=None,
        ppi=False,
        log_id=None,
    ):

    batch_size = trainer_args.batch_size
    read_scaler = trainer_args.read_scaler
    full = embedding_args.matrix_embed
    db_path = os.path.join(embedding_args.embedding_save_dir, f'{model_name}_{full}.db')

    if embedding_args.sql:
        if ppi:
            if full:
                raise ValueError('Full matrix embeddings not currently supported for SQL and PPI') # TODO: Implement
            DatasetClass = PairEmbedsLabelsDatasetFromDisk
            CollatorClass = PairEmbedsLabelsCollator
        else:
            DatasetClass = EmbedsLabelsDatasetFromDisk
            CollatorClass = EmbedsLabelsCollator
    else:
        if ppi:
            DatasetClass = PairEmbedsLabelsDataset
            CollatorClass = PairEmbedsLabelsCollator
        else:
            DatasetClass = EmbedsLabelsDataset
            CollatorClass = EmbedsLabelsCollator

    """
    For collator need to pass tokenizer, full, task_type
    For dataset need to pass
    hf_dataset, col_a, col_b, label_col, input_dim, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
    """

    data_collator = CollatorClass(tokenizer=tokenizer, full=full, task_type=task_type)
    train_dataset = DatasetClass(
        hf_dataset=train_dataset,
        input_dim=input_dim,
        task_type=task_type,
        db_path=db_path,
        emb_dict=emb_dict,
        batch_size=batch_size,
        read_scaler=read_scaler,
        full=full,
        train=True
    )
    valid_dataset = DatasetClass(
        hf_dataset=valid_dataset,
        input_dim=input_dim,
        task_type=task_type,
        db_path=db_path,
        emb_dict=emb_dict,
        batch_size=batch_size,
        read_scaler=read_scaler,
        full=full,
        train=False
    )
    test_dataset = DatasetClass(
        hf_dataset=test_dataset,
        input_dim=input_dim,
        task_type=task_type,
        db_path=db_path,
        emb_dict=emb_dict,
        batch_size=batch_size,
        read_scaler=read_scaler,
        full=full,
        train=False
    )

    return train(
        model=model,
        task_type=task_type,
        trainer_args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        model_name=model_name,
        data_name=data_name,
        log_id=log_id,
    )


def train_base_model(
        model,
        tokenizer,
        trainer_args,
        model_name,
        data_name,
        task_type,
        train_dataset,
        valid_dataset,
        test_dataset,
        ppi=False,
        log_id=None,
    ):
        if ppi:
            DatasetClass = PairStringLabelDataset
            CollatorClass = PairCollator_input_ids
        else:
            DatasetClass = StringLabelDataset
            CollatorClass = StringLabelsCollator

        data_collator = CollatorClass(tokenizer=tokenizer)

        train_dataset = DatasetClass(hf_dataset=train_dataset, train=True)
        valid_dataset = DatasetClass(hf_dataset=valid_dataset, train=False)
        test_dataset = DatasetClass(hf_dataset=test_dataset, train=False)

        return train(
            model=model,
            task_type=task_type,
            trainer_args=trainer_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            model_name=model_name,
            data_name=data_name,
            log_id=log_id,
        )



def train_hybrid_model(
        model,
        tokenizer,
        probe,
        trainer_args,
        embedding_args,
        model_name,
        data_name,
        tokenwise,
        input_dim,
        task_type,
        train_dataset,
        valid_dataset,
        test_dataset,
        emb_dict=None,
        ppi=False,
        log_id=None,
    ):
        probe = train_probe(
            model=probe,
            tokenizer=tokenizer,
            trainer_args=trainer_args,
            embedding_args=embedding_args,
            model_name=model_name,
            data_name=data_name,
            input_dim=input_dim,
            task_type=task_type,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            emb_dict=emb_dict,
            ppi=ppi,
            log_id=log_id,
        )
        config = HybridProbeConfig(
            tokenwise=tokenwise,
            pooling_types=embedding_args.embedding_pooling_types,
        )
        hybrid_model = HybridProbe(config=config, model=model, probe=probe)
        return train_base_model(
            model=hybrid_model,
            tokenizer=tokenizer,
            trainer_args=trainer_args,
            model_name=model_name,
            data_name=data_name,
            task_type=task_type,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            ppi=ppi,
            log_id=log_id,
        )
