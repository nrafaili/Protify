import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Tuple
from sklearn.utils import all_estimators
from lazy_predict import LazyRegressor, LazyClassifier, CLASSIFIERS, REGRESSORS
from ..metrics import get_dual_regression_scorer, get_dual_classification_scorer
from .scikit_hypers import HYPERPARAMETER_DISTRIBUTIONS


class TuningArguments:
    def __init__(
            self,
            n_iter: int = 100,
            cv: int = 3,
            random_state: int = 42,
            **kwargs,
    ):
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state


class SpecificModelArguments:
    def __init__(
        self,
        model_name: str,
        model_args: Dict[str, Any],
        production: bool = False,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_args = model_args
        self.production = production


class ModelResults:
    def __init__(
        self,
        initial_scores: pd.DataFrame,
        best_model_name: str,
        best_params: Dict[str, Any],
        final_scores: Dict[str, float],
        best_model: Any
    ):
        self.initial_scores = initial_scores
        self.best_model_name = best_model_name
        self.best_params = best_params
        self.final_scores = final_scores
        self.best_model = best_model

    def __str__(self) -> str:
        return (
            f"Best Model: {self.best_model_name}\n"
            f"Best Parameters: {self.best_params}\n"
            f"Final Scores: {self.final_scores}"
        )


def _tune_hyperparameters(
    model_class: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    custom_scorer: Any,
    n_iter: int = 100,
    cv: int = 3,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    """
    param_distributions = HYPERPARAMETER_DISTRIBUTIONS.get(model_name, {})
    if not param_distributions:
        return model_class(), {}

    random_search = RandomizedSearchCV(
        model_class(),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=custom_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def find_best_regressor(
    tuning_args: TuningArguments,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResults:
    """
    Find the best regression model through lazy prediction and hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        ModelResults object containing all results and the best model
    """
    # Initial lazy prediction
    regressor = LazyRegressor(
        verbose=4,
        ignore_warnings=True,
        custom_metric=get_dual_regression_scorer()
    )
    initial_scores = regressor.fit(X_train, X_test, y_train, y_test)
    
    # Get best model name and class
    best_model_name = initial_scores.index[0]
    best_model_class = regressor.models[best_model_name].named_steps['regressor'].__class__
    
    # Tune hyperparameters
    best_model, best_params = _tune_hyperparameters(
        best_model_class,
        best_model_name,
        X_train,
        y_train,
        get_dual_regression_scorer(),
        n_iter=tuning_args.n_iter,
        cv=tuning_args.cv,
        random_state=tuning_args.random_state
    )
    
    # Get final scores with tuned model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    final_scores = get_dual_regression_scorer()(best_model, X_test, y_test)
    
    return ModelResults(
        initial_scores=initial_scores,
        best_model_name=best_model_name,
        best_params=best_params,
        final_scores=final_scores,
        best_model=best_model
    )


def find_best_classifier(
    tuning_args: TuningArguments,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResults:
    """
    Find the best classification model through lazy prediction and hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        n_iter: Number of iterations for RandomizedSearchCV
        
    Returns:
        ModelResults object containing all results and the best model
    """
    # Initial lazy prediction
    classifier = LazyClassifier(
        verbose=4,
        ignore_warnings=True,
        custom_metric=get_dual_classification_scorer()
    )
    initial_scores = classifier.fit(X_train, X_test, y_train, y_test)
    
    # Get best model name and class
    best_model_name = initial_scores.index[0]
    best_model_class = classifier.models[best_model_name].named_steps['classifier'].__class__
    
    # Tune hyperparameters
    best_model, best_params = _tune_hyperparameters(
        best_model_class,
        best_model_name,
        X_train,
        y_train,
        get_dual_classification_scorer(),
        n_iter=tuning_args.n_iter,
        cv=tuning_args.cv,
        random_state=tuning_args.random_state
    )
    
    # Get final scores with tuned model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    final_scores = get_dual_classification_scorer()(best_model, X_test, y_test)
    
    return ModelResults(
        initial_scores=initial_scores,
        best_model_name=best_model_name,
        best_params=best_params,
        final_scores=final_scores,
        best_model=best_model
    )


def run_specific_model(
    model_args: SpecificModelArguments,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ModelResults:
    if model_args.production:
        X_train = np.concatenate([X_train, X_valid])
        y_train = np.concatenate([y_train, y_valid])

    model_name = model_args.model_name
    if model_name in CLASSIFIERS:
        scorer = get_dual_classification_scorer()
    elif model_name in REGRESSORS:
        scorer = get_dual_regression_scorer()
    else:
        raise ValueError(f"Model {model_name} not supported")

    model_list = all_estimators() # tuples (name, class)
    model_dict = {model[0]: model[1] for model in model_list}
    model_class = model_dict[model_name]
    cls = model_class(**model_args.model_args)
    cls.fit(X_train, y_train)
    final_scores = scorer(cls, X_test, y_test)
    return ModelResults(
        initial_scores=None,
        best_model_name=model_name,
        best_params=None,
        final_scores=final_scores,
        best_model=cls
    )
