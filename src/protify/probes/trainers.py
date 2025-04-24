import torch
import os
import numpy as np
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

from visualization.ci_plots import regression_ci_plot, classification_ci_plot


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


def train_model(
        trainer_args,
        embedding_args,
        model,
        model_name,
        data_name,
        input_dim,
        task_type,
        tokenizer,
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
    For dataset need to pass
    hf_dataset, col_a, col_b, label_col, input_dim, task_type, db_path, emb_dict, batch_size, read_scaler, full, train
    """
    if task_type == 'singlelabel':
        from metrics import compute_single_label_classification_metrics
        compute_metrics = compute_single_label_classification_metrics
    elif task_type == 'multilabel':
        from metrics import compute_multi_label_classification_metrics
        compute_metrics = compute_multi_label_classification_metrics
    elif task_type == 'regression':
        from metrics import compute_regression_metrics
        compute_metrics = compute_regression_metrics
    elif task_type == 'tokenwise':
        from metrics import compute_tokenwise_classification_metrics
        compute_metrics = compute_tokenwise_classification_metrics
    else:
        raise ValueError(f'Task type {task_type} not supported')


    data_collator = collate_builder(tokenizer=tokenizer, full=full, task_type=task_type)
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
    print(f'Initial metrics: \n{metrics}\n')

    trainer.train()

    valid_metrics = trainer.evaluate(valid_dataset)
    print(f'Final validation metrics: \n{valid_metrics}\n')

    y_pred, y_true, test_metrics = trainer.predict(test_dataset)
    print(f'y_pred: {y_pred.shape}')
    print(f'y_true: {y_true.shape}')
    print(f'Final test metrics: \n{test_metrics}\n')

    ### TODO PAUC plot
    if task_type == 'regression':
        regression_ci_plot(y_true, y_pred, trainer_args.plots_dir, data_name, model_name, log_id)
    elif task_type != 'multilabel':
        _, num_classes = y_pred.shape
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
        for class_idx in range(num_classes):
            y_pred_class = y_pred[:, class_idx]
            y_true_class = (y_true == class_idx).astype(int)
            classification_ci_plot(y_true_class, y_pred_class, trainer_args.plots_dir, data_name, model_name, log_id, class_idx)
    else:
        _, _, num_classes = y_pred.shape
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
        for class_idx in range(num_classes):
            y_pred_class = y_pred[:, :, class_idx].flatten()
            y_true_class = (y_true == class_idx).astype(int).flatten()
            classification_ci_plot(y_true_class, y_pred_class, trainer_args.plots_dir, data_name, model_name, log_id, class_idx)

    if trainer_args.save:
        try:
            trainer.model.push_to_hub(trainer_args.model_save_dir, private=True)
        except Exception as e:
            print(f'Error saving model: {e}')

    model = trainer.model.cpu()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()
    return model, valid_metrics, test_metrics
