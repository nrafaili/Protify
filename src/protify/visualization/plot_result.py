#!/usr/bin/env python3
"""
Create radar and bar plots for *all* datasets in a TSV.

Rules
-----
* Classification datasets → plot **MCC**  (fallback: F1, Accuracy)
* Regression    datasets → plot **R²**   (fallback: Spearman, Pearson)

The final plots therefore mix task types on the same axes.
Titles explicitly state that rule so readers know how to interpret numbers.
"""

from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_NAMES = {
    'Random': 'Random vectors',
    'Random-ESM2-8': r'$Random ESM2_{8M}$',
    'Random-ESM2-35': r'$Random ESM2_{35M}$',
    'Random-ESM2-150': r'$Random ESM2_{150M}$',
    'Random-ESM2-650': r'$Random ESM2_{650M}$',
    'Random-Transformer': 'Random Transformer',
    'ESM2-8': r'$ESM2_{8M}$',
    'ESM2-35': r'$ESM2_{35M}$',
    'ESM2-150': r'$ESM2_{150M}$',
    'ESM2-650': r'$ESM2_{650M}$',
    'ESM2-3B': r'$ESM2_{3B}$',
    'ESMC-300': r'$ESMC_{300M}$',
    'ESMC-600': r'$ESMC_{600M}$',
    'ESM2-diff-150': r'$ESMdiff_{150M}$',
    'ESM2-diffAV-150': r'$ESMdiffAV_{150M}$',
}

DATASET_NAMES = {
    'EC': 'EC',
    'GO-CC': r'$GO_{CC}$',
    'GO-BP': r'$GO_{BP}$',
    'GO-MF': r'$GO_{MF}$',
    'MB': 'MB',
    'DeepLoc-2': r'$DeepLoc_{2}$',
    'DeepLoc-10': r'$DeepLoc_{10}$',
    'enzyme-kcat': r'kcat',
    'solubility': 'solubility',
    'localization': 'localization',
    'temperature-stability': 'temperature stability',
    'peptide-HLA-MHC-affinity': 'peptide HLA MHC affinity',
    'optimal-temperature': 'optimal temperature',
    'optimal-ph': 'optimal pH',
    'material-production': 'material production',
    'fitness-prediction': 'fitness',
    'number-of-folds': 'folds',
    'cloning-clf': 'cloning',
    'stability-prediction': 'stability',
    'human-ppi': r'human PPI_{saprot}',
    'human-ppi-pinui': r'human PPI_{pinui}',
    'yest-ppi-pinui': r'yest PPI_{pinui}',
    'shs27-ppi': 'SHS27k',
    'shs148-ppi': 'SHS148k',
    'PPA-ppi': r'PPI affinity_{bindwell}',
    'synthyra-ppi': r'synthyra PPI_{bindwell}',
    'SecondaryStructure-3': r'SS_{3}',
    'SecondaryStructure-8': r'SS_{8}',
    'fluorescence-prediction': 'fluorescence',
    'plastic': 'plastic degradation',
    'gold-ppi': r'bernett_{PPI}',
}


CLS_PREFS: List[Tuple[str, str]] = [
    ("mcc",       "MCC"),
    ("f1",        "F1"),
    ("accuracy",  "Accuracy"),
]
REG_PREFS: List[Tuple[str, str]] = [
    ("r_squared", "R²"),
    ("spearman",  "Spearman ρ"),
    ("pearson",   "Pearson r"),
]


def is_regression(metrics: Dict[str, float]) -> bool:
    """Heuristic based on key names."""
    reg = ("spearman", "pearson", "r_squared", "rmse", "mse")
    cls = ("accuracy", "f1", "mcc", "auc", "precision", "recall")
    keys = {k.lower() for k in metrics}
    if any(k for k in keys if any(r in k for r in reg)):
        return True
    if any(k for k in keys if any(c in k for c in cls)):
        return False
    return False  # default to classification


def pick_metric(metrics: Dict[str, float], prefs: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Return (key, pretty_name) for the first preference present in metrics."""
    for k, nice in prefs:
        for mk in metrics:
            if mk.lower().endswith(k):
                return k, nice
    raise KeyError("No preferred metric found.")


def get_metric_value(metrics: Dict[str, float], key_suffix: str) -> float:
    """Fetch metric value case-/prefix-insensitively; NaN if absent."""
    for k, v in metrics.items():
        if k.lower().endswith(key_suffix):
            return v
    return math.nan


# ---------- plotting helpers -------------------------------------------------

def radar_factory(n_axes: int):
    theta = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return fig, ax, theta


def plot_radar(*,
               categories: List[str],
               models: List[str],
               data: List[List[float]],
               title: str,
               outfile: Path,
               normalize: bool = False):
    # Use pretty names for categories (datasets) and models
    pretty_categories = [DATASET_NAMES.get(cat, cat) for cat in categories]
    pretty_models = [MODEL_NAMES.get(m, m) for m in models]

    if normalize:
        arr = np.asarray(data)
        rng = np.where(arr.ptp(0) == 0, 1, arr.ptp(0))
        data = (arr - arr.min(0)) / rng
        # Convert back to list of lists for consistency
        data = data.tolist()

    # append mean column (do this after normalization if normalize=True)
    pretty_categories = pretty_categories + ["Avg"]
    data = [row + [np.nanmean(row)] for row in data]

    fig, ax, theta = radar_factory(len(pretty_categories))
    ax.set_thetagrids(np.degrees(theta), pretty_categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))

    palette = [plt.cm.tab20(i / len(pretty_models)) for i in range(len(pretty_models))]
    for i, (m, vals) in enumerate(zip(pretty_models, data)):
        ang = np.concatenate([theta, [theta[0]]])
        val = np.concatenate([vals,  [vals[0]]])
        ax.plot(ang, val, lw=2, label=m, color=palette[i])
        ax.fill(ang, val, alpha=.25, color=palette[i])

    ax.grid(True)
    plt.title(title, pad=20)
    plt.legend(bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches="tight")
    plt.close(fig)


def bar_plot(datasets: List[str],
             models: List[str],
             data: List[List[float]],
             metric_name: str,
             outfile: Path):
    rows = [
        {"Dataset": DATASET_NAMES.get(d, d), "Model": MODEL_NAMES.get(m, m), "Score": s}
        for m, col in zip(models, data)
        for d, s in zip(datasets, col)
    ]
    dfp = pd.DataFrame(rows)
    plt.figure(figsize=(max(12, .8 * len(datasets)), 8))
    sns.barplot(dfp, x="Dataset", y="Score", hue="Model")
    plt.title(f"{metric_name} across datasets (Cls→MCC, Reg→R²)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches="tight")
    plt.close()


def heatmap_plot(datasets: List[str],
                 models: List[str],
                 data: List[List[float]],
                 metric_name: str,
                 outfile: Path,
                 normalize: bool = False):
    arr = np.array(data).T  # shape: (num_datasets, num_models)
    # Compute average row (mean across datasets for each model)
    avg_row = np.nanmean(arr, axis=0, keepdims=True)
    arr_with_avg = np.vstack([arr, avg_row])
    datasets_plus_avg = datasets + ['Average']

    # Clean display names
    clean_model_names = [MODEL_NAMES.get(m, m) for m in models]
    clean_dataset_names = [DATASET_NAMES.get(d, d) for d in datasets_plus_avg]

    # Normalization (per row/dataset)
    if normalize:
        random_idx = None
        for i, m in enumerate(models):
            if m.lower() == 'random':
                random_idx = i
                break
        
        normalized_data = np.zeros_like(arr)
        
        # Normalize each dataset (row) independently
        for i in range(arr.shape[0]):
            if random_idx is not None:
                # Use random model as baseline if available
                random_performance = arr[i, random_idx]
                best_performance = np.nanmax(arr[i, :])
                denom = best_performance - random_performance
                denom = 1 if denom == 0 else denom
                normalized_data[i, :] = (arr[i, :] - random_performance) / denom
            else:
                # Fallback to min-max normalization if no Random model exists
                lowest_performance = np.nanmin(arr[i, :])
                best_performance = np.nanmax(arr[i, :])
                denom = best_performance - lowest_performance
                denom = 1 if denom == 0 else denom
                normalized_data[i, :] = (arr[i, :] - lowest_performance) / denom
        
        # Add average row to normalized data
        avg_row_norm = np.nanmean(normalized_data, axis=0, keepdims=True)
        plot_arr = np.vstack([normalized_data, avg_row_norm])
        mask = np.abs(plot_arr - 1.0) < 0.005
        cmap = 'coolwarm'
        center = 0.8
        vmin = 0.6
        vmax = 1.0
        cbar_label = 'Normalized Performance'
    else:
        plot_arr = arr_with_avg
        mask = np.zeros_like(plot_arr, dtype=bool)
        cmap = 'coolwarm'
        center = None
        vmin = None
        vmax = None
        cbar_label = metric_name

    plt.figure(figsize=(max(15, .8 * len(clean_model_names)), max(10, .5 * len(clean_dataset_names))))
    ax = sns.heatmap(plot_arr,  # rows: datasets, cols: models
                     xticklabels=clean_model_names,
                     yticklabels=clean_dataset_names,
                     cmap=cmap,
                     center=center,
                     vmin=vmin,
                     vmax=vmax,
                     annot=True,
                     fmt='.2f',
                     annot_kws={'size': 8},
                     cbar_kws={'label': cbar_label})
    for i in range(plot_arr.shape[0]):
        for j in range(plot_arr.shape[1]):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    plt.title(f'{cbar_label} heatmap (Cls→MCC, Reg→R²)', pad=20)
    plt.ylabel('Dataset')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=450, bbox_inches='tight')
    plt.close()


# ---------- main entry -------------------------------------------------------

def load_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    for c in df.columns:
        if c != "dataset":
            df[c] = df[c].apply(json.loads)
    return df


def create_plots(tsv: str, outdir: str):
    tsv, outdir = Path(tsv), Path(outdir)
    df = load_tsv(tsv)
    models = [c for c in df.columns if c != "dataset"]

    # Resolve metric per-dataset (MCC or R², w/ fallbacks).
    datasets, scores_by_model = [], {m: [] for m in models}

    for _, row in df.iterrows():
        name = row["dataset"]
        metrics0 = row[models[0]]
        task = "regression" if is_regression(metrics0) else "classification"
        prefs = REG_PREFS if task == "regression" else CLS_PREFS

        try:
            suffix, pretty = pick_metric(metrics0, prefs)
        except KeyError:
            print(f"[WARN] {name}: no suitable metric – skipped.")
            continue

        datasets.append(name)
        for m in models:
            val = get_metric_value(row[m], suffix)
            scores_by_model[m].append(val)

    if not datasets:
        raise RuntimeError("No plottable datasets found.")

    # assemble lists in model order
    plot_matrix = [scores_by_model[m] for m in models]

    # Sort models by average score (ascending: worst to best)
    model_avgs = [np.nanmean(scores) for scores in plot_matrix]
    sorted_indices = np.argsort(model_avgs)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_plot_matrix = [plot_matrix[i] for i in sorted_indices]

    fig_tag = tsv.stem
    outdir = outdir / fig_tag
    outdir.mkdir(parents=True, exist_ok=True)

    # File paths for all plot types
    radar_path = outdir / f"{fig_tag}_radar_all.png"
    radar_path_norm = outdir / f"{fig_tag}_radar_all_normalized.png"
    bar_path = outdir / f"{fig_tag}_bar_all.png"
    bar_path_norm = outdir / f"{fig_tag}_bar_all_normalized.png"
    heatmap_path = outdir / f"{fig_tag}_heatmap_all.png"
    heatmap_path_norm = outdir / f"{fig_tag}_heatmap_all_normalized.png"

    subtitle = "Classification datasets plot MCC; Regression datasets plot R²"
    # Radar plot keeps original order
    plot_radar(categories=datasets,
               models=models,
               data=plot_matrix,
               title=subtitle,
               outfile=radar_path,
               normalize=False)
    plot_radar(categories=datasets,
               models=models,
               data=plot_matrix,
               title=subtitle + " (Normalized)",
               outfile=radar_path_norm,
               normalize=True)
    # Bar and heatmap use sorted order
    bar_plot(datasets, sorted_models, sorted_plot_matrix, "Score (MCC / R²)", bar_path)
    # Normalized bar plot
    # For bar plot normalization, use min-max per dataset (column-wise normalization)
    arr = np.asarray(sorted_plot_matrix)
    rng = np.where(arr.ptp(0) == 0, 1, arr.ptp(0))
    arr_norm = (arr - arr.min(0)) / rng
    bar_plot(datasets, sorted_models, arr_norm.tolist(), "Score (MCC / R²) (Normalized)", bar_path_norm)
    # Heatmap
    heatmap_plot(datasets, sorted_models, sorted_plot_matrix, "Score (MCC / R²)", heatmap_path, normalize=False)
    heatmap_plot(datasets, sorted_models, sorted_plot_matrix, "Score (MCC / R²)", heatmap_path_norm, normalize=True)

    print(f"Radar saved to {radar_path}")
    print(f"Radar (normalized) saved to {radar_path_norm}")
    print(f"Bar   saved to {bar_path}")
    print(f"Bar (normalized) saved to {bar_path_norm}")
    print(f"Heatmap saved to {heatmap_path}")
    print(f"Heatmap (normalized) saved to {heatmap_path_norm}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate radar, bar, and heatmap plots for all datasets. Always saves both normalized and unnormalized versions.")
    ap.add_argument("--input", required=True, help="TSV file with metrics")
    ap.add_argument("--output_dir", default="plots", help="Directory for plots")
    args = ap.parse_args()

    create_plots(Path(args.input), Path(args.output_dir))
    print("Finished.")


if __name__ == "__main__":
    # py -m visualization.plot_result

    # --- TESTS FOR PLOTTING FUNCTIONS ---
    print("\nRunning plot function tests...")
    from pathlib import Path
    tmpdir = Path("plots/test_plots")
    tmpdir.mkdir(parents=True, exist_ok=True)
    # Dummy data
    categories = ["A", "B", "C"]
    models = ["Model1", "Model2"]
    data = [
        [0.8, 0.6, 0.7],
        [0.5, 0.9, 0.4],
    ]
    # Radar plot
    radar_path = tmpdir / "test_radar.png"
    plot_radar(categories=categories, models=models, data=data, title="Test Radar", outfile=radar_path)
    assert radar_path.exists(), "Radar plot not created!"
    print(f"Radar plot test passed: {radar_path}")
    # Normalized radar plot
    radar_path_norm = tmpdir / "test_radar_normalized.png"
    plot_radar(categories=categories, models=models, data=data, title="Test Radar (Normalized)", outfile=radar_path_norm, normalize=True)
    assert radar_path_norm.exists(), "Normalized radar plot not created!"
    print(f"Normalized radar plot test passed: {radar_path_norm}")
    # Bar plot
    bar_path = tmpdir / "test_bar.png"
    bar_plot(categories, models, data, "Test Metric", bar_path)
    assert bar_path.exists(), "Bar plot not created!"
    print(f"Bar plot test passed: {bar_path}")
    # Normalized bar plot
    arr = np.asarray(data)
    rng = np.where(arr.ptp(0) == 0, 1, arr.ptp(0))
    arr_norm = (arr - arr.min(0)) / rng
    bar_path_norm = tmpdir / "test_bar_normalized.png"
    bar_plot(categories, models, arr_norm.tolist(), "Test Metric (Normalized)", bar_path_norm)
    assert bar_path_norm.exists(), "Normalized bar plot not created!"
    print(f"Normalized bar plot test passed: {bar_path_norm}")
    # Heatmap plot
    heatmap_path = tmpdir / "test_heatmap.png"
    heatmap_plot(categories, models, data, "Test Metric", heatmap_path)
    assert heatmap_path.exists(), "Heatmap plot not created!"
    print(f"Heatmap plot test passed: {heatmap_path}")
    # Normalized heatmap plot
    heatmap_path_norm = tmpdir / "test_heatmap_normalized.png"
    heatmap_plot(categories, models, data, "Test Metric", heatmap_path_norm, normalize=True)
    assert heatmap_path_norm.exists(), "Normalized heatmap plot not created!"
    print(f"Normalized heatmap plot test passed: {heatmap_path_norm}")
    print("All plot function tests passed!\n")
