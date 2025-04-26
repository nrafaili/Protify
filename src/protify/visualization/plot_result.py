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


# ---------- metric logic -----------------------------------------------------

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
    if normalize:
        arr = np.asarray(data)
        rng = np.where(arr.ptp(0) == 0, 1, arr.ptp(0))
        data = (arr - arr.min(0)) / rng

    # append mean column
    categories = categories + ["Avg"]
    data = [row + [np.nanmean(row)] for row in data]

    fig, ax, theta = radar_factory(len(categories))
    ax.set_thetagrids(np.degrees(theta), categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))

    palette = [plt.cm.tab20(i / len(models)) for i in range(len(models))]
    for i, (m, vals) in enumerate(zip(models, data)):
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
        {"Dataset": d, "Model": m, "Score": s}
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


# ---------- main entry -------------------------------------------------------

def load_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    for c in df.columns:
        if c != "dataset":
            df[c] = df[c].apply(json.loads)
    return df


def create_plots(tsv: str, outdir: str, normalize: bool = False):
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
    fig_tag = tsv.stem
    outdir = outdir / fig_tag
    outdir.mkdir(parents=True, exist_ok=True)

    radar_path = outdir / f"{fig_tag}_radar_all.png"
    bar_path   = outdir / f"{fig_tag}_bar_all.png"

    subtitle = "Classification datasets plot MCC; Regression datasets plot R²"
    plot_radar(categories=datasets,
               models=models,
               data=plot_matrix,
               title=subtitle,
               outfile=radar_path,
               normalize=normalize)
    bar_plot(datasets, models, plot_matrix, "Score (MCC / R²)", bar_path)

    print(f"✓ Radar saved to {radar_path}")
    print(f"✓ Bar   saved to {bar_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate radar & bar plots for all datasets.")
    ap.add_argument("--input", required=True, help="TSV file with metrics")
    ap.add_argument("--output_dir", default="plots", help="Directory for plots")
    ap.add_argument("--normalize", action="store_true",
                    help="Min-max normalise scores per dataset before plotting")
    args = ap.parse_args()

    create_plots(Path(args.input), Path(args.output_dir), args.normalize)
    print("Finished.")


if __name__ == "__main__":
    main()
