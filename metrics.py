from sklearn.metrics import (
    make_scorer,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    auc,
    matthews_corrcoef,
)

from scipy.stats import pearsonr, spearmanr


### Scikit-learn
def get_pearson_scorer():
    def pearson_r(y_true, y_pred):
        return pearsonr(y_true, y_pred).correlation
    return make_scorer(pearson_r, greater_is_better=True)


def get_spearman_scorer():
    def spearman_r(y_true, y_pred):
        return spearmanr(y_true, y_pred).correlation
    return make_scorer(spearman_r, greater_is_better=True)


def get_r2_scorer():
    return make_scorer(r2_score, greater_is_better=True)


def get_dual_regression_scorer():
    def dual_score(y_true, y_pred):
        return spearmanr(y_true, y_pred).correlation * r2_score(y_true, y_pred)
    return make_scorer(dual_score, greater_is_better=True)


def get_dual_classification_scorer():
    def dual_score(y_true, y_pred):
        return f1_score(y_true, y_pred) * matthews_corrcoef(y_true, y_pred)
    return make_scorer(dual_score, greater_is_better=True)


### PyTorch / Transformers
def compute_single_label_metrics(y_true, y_pred):
    pass


def compute_multi_label_metrics(y_true, y_pred):
    pass


def compute_tokenwise_metrics(y_true, y_pred):
    pass


def compute_regression_metrics(y_true, y_pred):
    pass
