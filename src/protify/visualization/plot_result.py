import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sns
from pathlib import Path


def radar_factory(num_vars):
    """Create radar chart setup with the given number of variables."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(10, 10))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return fig, ax, theta


def plot_radar(categories, models, scores, colors=None, normalize=False, average=False, title=None, output_file=None):
    """
    Create a radar plot with the given categories, models, and scores.
    
    Parameters:
    - categories: List of category names
    - models: List of model names
    - scores: List of lists, where each inner list contains scores for a model across all categories
    - colors: Optional list of colors for each model
    - normalize: Whether to normalize scores across models for each category
    - average: Whether to add an "Avg" category with the average score for each model
    - title: Optional title for the plot
    - output_file: Path to save the figure
    """
    # If average flag is True, add an extra "Avg" category computed as the mean of the others.
    if average:
        # Append "Avg" to the list of categories.
        categories = categories + ["Avg"]
        # For each model, compute the average of the original scores and append it.
        scores = [score + [np.mean(score)] for score in scores]
    
    num_vars = len(categories)
    fig, ax, theta = radar_factory(num_vars)

    # If no colors specified, pick from a colormap with many distinct colors.
    if colors is None:
        colors = [plt.cm.tab20(i / len(models)) for i in range(len(models))]

    ax.set_thetagrids(np.degrees(theta), categories, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # Normalize the scores if required (after computing the average).
    if normalize:
        scores_arr = np.array(scores)
        s_min = scores_arr.min(axis=0)
        s_max = scores_arr.max(axis=0)
        diff = s_max - s_min
        # Prevent division by zero in case a category has constant scores.
        diff[diff == 0] = 1
        scores = (scores_arr - s_min) / diff

    for i, model_name in enumerate(models):
        # Convert the model's scores to a NumPy array.
        values = np.array(scores[i])
        # Close the polygon by appending the first value at the end.
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((theta, [theta[0]]))
        
        ax.plot(angles, values, color=colors[i], label=model_name, linewidth=2, zorder=1)
        ax.fill(angles, values, color=colors[i], alpha=0.25, zorder=1)

    ax.grid(True, zorder=0)
    # Set higher zorder for tick labels and lines to appear on top
    for label in ax.yaxis.get_ticklabels():
        label.set_zorder(5)
    ax.yaxis.set_tick_params(zorder=5)
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=14, pad=20)
    
    # Place the legend outside and to the right of the plot.
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    
    # Save the figure if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=450, bbox_inches='tight')
    
    plt.close()


def load_data(tsv_file):
    """Load data from TSV file and parse the JSON strings into dictionaries."""
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Convert JSON strings to dictionaries
    for col in df.columns:
        if col != 'dataset':
            df[col] = df[col].apply(json.loads)
    
    return df


def is_regression_task(data_row):
    """Determine if a dataset is a regression task based on available metrics."""
    # Check any model's metrics to determine task type
    for col in data_row.index:
        if col == 'dataset':
            continue
        
        metrics = data_row[col]
        if isinstance(metrics, dict):
            # Regression tasks typically have r_squared, spearman_rho, etc.
            if 'eval_spearman_rho' in metrics or 'eval_r_squared' in metrics:
                return True
            # Classification tasks have metrics like accuracy, f1, etc.
            elif 'eval_accuracy' in metrics or 'eval_f1' in metrics or 'eval_mcc' in metrics:
                return False
    
    # Default to classification if can't determine
    return False


def create_summary_plots(df, output_dir, normalize=False):
    """Create summary radar plots for classification and regression datasets."""
    # Get all model names
    models = [col for col in df.columns if col != 'dataset']
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group datasets by task type
    classification_datasets = []
    regression_datasets = []
    
    for idx, row in df.iterrows():
        dataset_name = row['dataset']
        is_regression = is_regression_task(row)
        
        if is_regression:
            regression_datasets.append(dataset_name)
        else:
            classification_datasets.append(dataset_name)
    
    # Create summary plot for classification datasets (MCC)
    if classification_datasets:
        create_task_summary_plot(
            df, 
            dataset_names=classification_datasets,
            metric='eval_mcc',
            title_metric="MCC",
            output_dir=output_dir,
            normalize=normalize
        )
    
    # Create summary plot for regression datasets (Spearman's rho)
    if regression_datasets:
        create_task_summary_plot(
            df, 
            dataset_names=regression_datasets,
            metric='eval_spearman_rho',
            title_metric="Spearman's rho",
            output_dir=output_dir,
            normalize=normalize
        )


def create_task_summary_plot(df, dataset_names, metric, title_metric, output_dir, normalize=False):
    """Create a summary radar plot for datasets of the same task type."""
    # Get all model names
    models = [col for col in df.columns if col != 'dataset']
    
    # Get scores for each model across all selected datasets
    all_scores = {model: [] for model in models}
    valid_datasets = []
    
    for dataset in dataset_names:
        row = df[df['dataset'] == dataset].iloc[0]
        
        # Check if all models have this metric for this dataset
        valid_metric = True
        for model in models:
            try:
                value = row[model].get(metric, np.nan)
                if math.isnan(value):
                    valid_metric = False
                    break
            except (KeyError, AttributeError, TypeError):
                valid_metric = False
                break
        
        if valid_metric:
            valid_datasets.append(dataset)
            for model in models:
                all_scores[model].append(row[model][metric])
    
    if not valid_datasets:
        print(f"  No datasets have {metric} values for all models, skipping summary plot...")
        return
    
    # Prepare data for the radar plot
    categories = valid_datasets
    plot_models = models
    
    # Create a score list for each model
    scores = [all_scores[model] for model in plot_models]
    
    # Set colors for the models (using tab20 colormap)
    colors = [plt.cm.tab20(i / len(plot_models)) for i in range(len(plot_models))]
    
    # Create the radar plot
    task_type = "regression" if metric == 'eval_spearman_rho' else "classification"
    output_file = Path(output_dir) / f'summary_radar_{task_type}.png'
    title = f"Summary of {title_metric} values across {task_type} datasets"
    
    plot_radar(
        categories=categories,
        models=plot_models,
        scores=scores,
        colors=colors,
        normalize=normalize,
        average=True,
        title=title,
        output_file=output_file
    )
    
    print(f"Created summary {task_type} radar plot at {output_file}")
    
    # Now create a comprehensive bar plot for the same data
    create_bar_plot(
        valid_datasets, 
        plot_models, 
        scores, 
        task_type, 
        title_metric, 
        output_dir
    )


def create_bar_plot(datasets, models, scores, task_type, metric_name, output_dir):
    """Create a comprehensive bar plot with all datasets and models."""
    # Prepare data for the bar plot
    data = []
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            data.append({
                'Model': model,
                'Dataset': dataset,
                'Score': scores[i][j]
            })
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(max(12, len(datasets) * 0.8), 8))
    
    # Use seaborn for better visuals
    sns.barplot(x='Dataset', y='Score', hue='Model', data=df_plot)
    
    # Customize the plot
    plt.title(f'{metric_name} comparison across {task_type} datasets', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    output_file = Path(output_dir) / f'bar_plot_{task_type}.png'
    plt.savefig(output_file, dpi=450, bbox_inches='tight')
    plt.close()
    
    print(f"Created {task_type} bar plot at {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create summary radar and bar plots from TSV results file')
    parser.add_argument('--input', type=str, required=True, help='Path to TSV file with results')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--normalize', action='store_true', help='Normalize scores across models for each category')
    
    args = parser.parse_args()
    
    # Load and process data
    df = load_data(args.input)
    
    # Create summary plots
    create_summary_plots(df, args.output_dir, args.normalize)
    
    print(f"All plots have been saved to {args.output_dir}")


if __name__ == "__main__":
    main()
