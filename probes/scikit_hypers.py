HYPERPARAMETER_DISTRIBUTIONS = {
    ### Classification Models ###
    "RandomForestClassifier": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBClassifier": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "LGBMClassifier": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "BaggingClassifier": {
        "n_estimators": [10, 20, 30, 40, 50],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
    },
    "ExtraTreesClassifier": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVC": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.1, 0.01],
    },
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs", "newton-cg", "sag"],
        "max_iter": [100, 200, 300, 500],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },
    "BernoulliNB": {
        "alpha": [0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    },
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "LinearDiscriminantAnalysis": {
        "solver": ["svd", "lsqr", "eigen"],
        "shrinkage": [None, "auto", 0.1, 0.5, 0.9],
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "store_covariance": [True, False],
    },
    "RidgeClassifier": {
        "alpha": [0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "sag", "saga"],
    },
    "PassiveAggressiveClassifier": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "max_iter": [100, 500, 1000],
        "early_stopping": [True, False],
        "validation_fraction": [0.1, 0.2],
    },
    "NearestCentroid": {
        "metric": ["euclidean", "manhattan"],
        "shrink_threshold": [None, 0.1, 0.5, 0.9],
    },
    "NuSVC": {
        "nu": [0.1, 0.3, 0.5, 0.7, 0.9],
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "gamma": ["scale", "auto", 0.1, 0.01],
    },
    "LabelPropagation": {
        "kernel": ["knn", "rbf"],
        "gamma": [20, 50, 100],
        "n_neighbors": [3, 5, 7],
    },
    "LabelSpreading": {
        "kernel": ["knn", "rbf"],
        "gamma": [20, 50, 100],
        "n_neighbors": [3, 5, 7],
        "alpha": [0.1, 0.3, 0.5],
    },
    "DummyClassifier": {
        "strategy": ["stratified", "most_frequent", "prior", "uniform"],
    },
    "DummyRegressor": {
        "strategy": ["mean", "median", "quantile", "constant"],
    },

    ### Regression Models ###
    "RandomForestRegressor": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 5, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBRegressor": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "LGBMRegressor": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "AdaBoostRegressor": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1, 0.15],
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "BaggingRegressor": {
        "n_estimators": [10, 20, 30, 40, 50],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
    },
    "ExtraTreesRegressor": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVR": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.1, 0.01],
    },
    "LinearRegression": {
        "fit_intercept": [True, False],
        "positive": [True, False],
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "sag", "saga"],
    },
    "Lasso": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "selection": ["cyclic", "random"],
    },
    "ElasticNet": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "fit_intercept": [True, False],
        "selection": ["cyclic", "random"],
    },
    "Lars": {
        "fit_intercept": [True, False],
        "normalize": [True, False],
        "n_nonzero_coefs": [100, 300, 500, "auto"],
    },
    "BayesianRidge": {
        "n_iter": [100, 300, 500],
        "alpha_1": [1e-6, 1e-5, 1e-4],
        "alpha_2": [1e-6, 1e-5, 1e-4],
        "lambda_1": [1e-6, 1e-5, 1e-4],
        "lambda_2": [1e-6, 1e-5, 1e-4],
    },
    "HuberRegressor": {
        "epsilon": [1.1, 1.35, 1.5, 2.0],
        "max_iter": [100, 200, 300],
        "alpha": [0.0001, 0.001, 0.01],
    },
    "PassiveAggressiveRegressor": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "max_iter": [100, 500, 1000],
        "early_stopping": [True, False],
        "validation_fraction": [0.1, 0.2],
    },
    "RANSACRegressor": {
        "min_samples": [0.1, 0.5, 0.9],
        "max_trials": [100, 200],
        "max_skips": [100, 200],
        "stop_n_inliers": [100, 200],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    },
    "KernelRidge": {
        "alpha": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "gamma": [0.1, 0.5, 1.0, "scale"],
    },
}