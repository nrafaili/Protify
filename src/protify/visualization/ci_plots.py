import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from pauc import plot_roc_with_ci


def regression_ci_plot(y_true, y_pred, output_dir, data_name, model_name, log_id):
    """
    Calculate the spearman rho and p-value of the regression model.
    Plot the line of best fit with 95% confidence intervals for spearman rho.
    Display the R-squared value, spearman rho, pearson rho, and p-values.
    """
    # Compute R‑squared, Spearman and Pearson correlations
    r2 = r2_score(y_true, y_pred)
    r_s, p_s = spearmanr(y_true, y_pred)
    r_p, p_p = pearsonr(y_true, y_pred)

    # Create scatter plot and regression line with 95% CI
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    sns.regplot(
        x=y_true, y=y_pred,
        ci=95, ax=ax, scatter=False,
        line_kws={'color': 'red'}
    )

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Regression Plot with 95% Confidence Interval')

    # Annotate statistics on the plot
    stats_text = (
        f"$R^2$ = {r2:.2f}\n"
        f"Spearman $\\rho$ = {r_s:.2f}  (p = {p_s:.2e})\n"
        f"Pearson $\\rho$ = {r_p:.2f}  (p = {p_p:.2e})"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=12, verticalalignment='top'
    )

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_regression_ci_plot.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def classification_ci_plot(y_true, y_pred, output_dir, data_name, model_name, log_id, current_class):
    """
    Use pauc to display classification plot
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{data_name}_{model_name}_{log_id}_classification_ci_plot_{current_class}.png")
    plot_roc_with_ci(y_true, y_pred, save_path)


if __name__ == "__main__":
    # test the function
    #y_true = np.random.rand(100)
    #y_pred = np.random.rand(100)
    output_dir = "plots"
    #regression_ci_plot(y_true, y_pred, output_dir, "regression", "regression", "regression")

    # binary classification
    y_true = np.random.randint(0, 2, (100, ))
    y_pred = np.random.rand(100, 2)
    # softmax y_pred
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
    num_examples, num_classes = y_pred.shape
    for class_idx in range(num_classes):
        y_pred_class = y_pred[:, class_idx]
        y_true_class = (y_true == class_idx).astype(int)
        classification_ci_plot(y_true_class, y_pred_class, output_dir, "binary", "binary", "binary", class_idx)

    # multi-class classification
    y_true = np.random.randint(0, 3, (100, ))
    y_pred = np.random.rand(100, 3)
    # softmax y_pred
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
    num_examples, num_classes = y_pred.shape
    for class_idx in range(num_classes):
        y_pred_class = y_pred[:, class_idx]
        y_true_class = (y_true == class_idx).astype(int)
        classification_ci_plot(y_true_class, y_pred_class, output_dir, "multiclass", "multiclass", "multiclass", class_idx)

    # multi-label classification
    y_true = np.random.randint(0, 2, (100, 100))
    y_pred = np.random.rand(100, 100, 2)
    # softmax y_pred
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
    num_examples, seq_len, num_classes = y_pred.shape
    for class_idx in range(num_classes):
        y_pred_class = y_pred[:, :, class_idx].flatten()
        y_true_class = (y_true == class_idx).astype(int).flatten()
        print(y_pred_class.shape, y_true_class.shape)
        classification_ci_plot(y_true_class, y_pred_class, output_dir, "multilabel", "multilabel", "multilabel", class_idx)

