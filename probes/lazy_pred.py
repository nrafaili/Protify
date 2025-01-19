from lazypredict.Supervised import LazyRegressor, LazyClassifier
from ..metrics import get_dual_regression_scorer, get_dual_classification_scorer


def find_best_regressor(X_train, y_train, X_test, y_test):
    regressor = LazyRegressor(verbose=1, ignore_warnings=True, custom_metric=get_dual_regression_scorer())
    scores = regressor.fit(X_train, X_test, y_train, y_test)
    # scores.sort
    print(scores)
    return scores


def find_best_classifier(X_train, y_train, X_test, y_test):
    classifier = LazyClassifier(verbose=1, ignore_warnings=True, custom_metric=get_dual_classification_scorer())
    scores = classifier.fit(X_train, X_test, y_train, y_test)
    print(scores)
    return scores
