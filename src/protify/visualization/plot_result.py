#!/usr/bin/env python3
"""
Create summary radar and bar plots from a TSV produced by our training
pipeline.  Each non-“dataset” column is a JSON blob of metrics, e.g.

dataset   ModelA                                             ModelB
DL10_reg  '{"test_loss":0.81,"test_f1":0.57,"test_mcc":0.69}' ...

The script figures out whether every dataset is classification or regression
and then picks the best common metric to compare (MCC / F1 / Accuracy or
Spearman / R² / Pearson).
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def radar_factory(num_vars: int):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return fig, ax, theta


def plot_radar(*,
               categories: List[str],
               models: List[str],
               scores: List[List[float]],
               title: str,
               output_file: Path,
               colors: Optional[List] = None,
               normalize: bool = False,
               average: bool = True):
    if average:
        categories = categories + ["Avg"]
        scores = [score + [np.mean(score)] for score in scores]

    fig, ax, theta = radar_factory(len(categories))

    if colors is None:
        colors = [plt.cm.tab20(i / len(models)) for i in range(len(models))]

    ax.set_thetagrids(np.degrees(theta), categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    if normalize:
        arr = np.asarray(scores)
        s_min, s_max = arr.min(0), arr.max(0)
        rng = np.where(s_max - s_min == 0, 1, s_max - s_min)
        scores = (arr - s_min) / rng

    for i, (model, vals) in enumerate(zip(models, scores)):
        vals = np.concatenate((vals, [vals[0]]))
        angs = np.concatenate((theta, [theta[0]]))
        ax.plot(angs, vals, color=colors[i], label=model, lw=2)
        ax.fill(angs, vals, color=colors[i], alpha=0.25)

    ax.grid(True)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_file, dpi=450, bbox_inches='tight')
    plt.close()


def bar_plot(datasets: List[str],
             models: List[str],
             scores: List[List[float]],
             *,
             metric_name: str,
             task_type: str,
             output_file: Path):
    rows = [{'Dataset': d, 'Model': m, 'Score': s}
            for m, col in zip(models, scores)
            for d, s in zip(datasets, col)]
    df_plot = pd.DataFrame(rows)

    plt.figure(figsize=(max(12, .8 * len(datasets)), 8))
    sns.barplot(x='Dataset', y='Score', hue='Model', data=df_plot)
    plt.title(f'{metric_name} comparison across {task_type} datasets')
    plt.xlabel('Dataset')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, dpi=450, bbox_inches='tight')
    plt.close()


def load_data(tsv: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep='\t')
    for col in df.columns:
        if col != 'dataset':
            df[col] = df[col].apply(json.loads)
    return df


def is_regression(metrics: Dict[str, float]) -> bool:
    """Heuristic: look for any rank/continuous metrics."""
    reg_keys = ('spearman', 'pearson', 'r_squared', 'rmse', 'mse')
    cls_keys = ('accuracy', 'f1', 'mcc', 'auc', 'precision', 'recall')
    keys = [k.lower() for k in metrics.keys()]
    if any(k for k in keys if any(r in k for r in reg_keys)):
        return True
    if any(k for k in keys if any(c in k for c in cls_keys)):
        return False
    # default: assume classification
    return False


CLASSIFICATION_METRICS = [
    # preference order
    ('mcc', 'Matthews correlation'),
    ('f1', 'F1'),
    ('accuracy', 'Accuracy'),
]

REGRESSION_METRICS = [
    ('spearman', "Spearman's ρ"),
    ('r_squared', 'R²'),
    ('pearson', "Pearson's r"),
]


def choose_metric(rows: List[pd.Series],
                  models: List[str],
                  metric_table: List):
    """
    Pick the first metric that every model possesses for all supplied rows.
    Returns (metric_key, pretty_name) or (None, None).
    """
    for short_key, pretty in metric_table:
        for row in rows:
            for m in models:
                metrics = row[m]
                # Search regardless of 'test_' / 'eval_' prefix, case-insensitive
                cand = next((v for k, v in metrics.items()
                             if k.lower().endswith(short_key)), None)
                if cand is None or math.isnan(cand):
                    break
            else:   # inner loop ok
                continue
            break   # some missing
        else:       # every dataset & model had it
            return short_key, pretty
    return None, None


def extract_metric_value(metrics: Dict[str, float], metric_key: str):
    # pull key regardless of prefix/case
    for k, v in metrics.items():
        if k.lower().endswith(metric_key):
            return v
    return np.nan


def create_plots(df: pd.DataFrame, *,
                 outdir: Path,
                 fig_id: str,
                 normalize: bool = False):

    models = [c for c in df.columns if c != 'dataset']
    outdir.mkdir(parents=True, exist_ok=True)

    # split datasets by task
    cls_rows, reg_rows = [], []
    for _, row in df.iterrows():
        first_metrics = row[models[0]]
        (reg_rows if is_regression(first_metrics) else cls_rows).append(row)

    for task_rows, metric_table, task_name in [
        (cls_rows, CLASSIFICATION_METRICS, 'classification'),
        (reg_rows, REGRESSION_METRICS, 'regression'),
    ]:
        if not task_rows:
            continue

        metric_key, metric_pretty = choose_metric(task_rows, models, metric_table)
        if metric_key is None:
            print(f'No common metric for {task_name}; skipping.')
            continue

        datasets = [r['dataset'] for r in task_rows]
        scores_by_model = []
        for m in models:
            scores_by_model.append([
                extract_metric_value(r[m], metric_key) for r in task_rows
            ])

        # radar
        radar_path = outdir / f'{fig_id}_radar_{task_name}.png'
        plot_radar(categories=datasets,
                   models=models,
                   scores=scores_by_model,
                   title=f'{metric_pretty} across {task_name} datasets',
                   output_file=radar_path,
                   normalize=normalize)
        print(f'🌐  radar plot → {radar_path}')

        # bar
        bar_path = outdir / f'{fig_id}_bar_{task_name}.png'
        bar_plot(datasets, models, scores_by_model,
                 metric_name=metric_pretty,
                 task_type=task_name,
                 output_file=bar_path)
        print(f'📊  bar plot  → {bar_path}')


def main():
    p = argparse.ArgumentParser(description='Generate radar/bar plots from TSV')
    p.add_argument('--input', required=True, help='TSV file with metrics')
    p.add_argument('--output_dir', default='plots')
    p.add_argument('--normalize', action='store_true',
                   help='Normalize scores per category')
    args = p.parse_args()

    df = load_data(args.input)
    fig_id = Path(args.input).stem  # e.g. 2025-04-24-15-50_SECS
    create_plots(df,
                 outdir=Path(args.output_dir),
                 fig_id=fig_id,
                 normalize=args.normalize)
    print('✅  All plots saved.')


if __name__ == '__main__':
    main()
