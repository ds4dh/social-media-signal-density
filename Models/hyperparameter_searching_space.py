

import optuna

from constant import N_JOBS


def build_hyperparameter_search_space(
    param,
    trial: optuna.trial.Trial,
    base_ratio: float | None,
) -> dict:
    """
    Hyperparameter search space for XGBoost on a small dataset (~1100 samples).

    Supports both binary classification (base_ratio provided) and regression
    (base_ratio=None). The objective, eval_metric, and class-imbalance
    parameters are set automatically based on the task.

    Parameters
    ----------
    param : argparse.Namespace
        General config. Uses param.random_seed for reproducibility.
    trial : optuna.trial.Trial
        Optuna trial object.
    base_ratio : float or None
        Ratio of negative to positive samples for classification.
        Pass None for regression tasks.

    Returns
    -------
    dict
        Hyperparameter dictionary to pass to XGBClassifier or XGBRegressor.
    """
    is_regression = base_ratio is None

    params = {
        # --- Core tree structure ---
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        # --- Stochastic regularization ---
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        # --- Explicit regularization ---
        "gamma":            trial.suggest_float("gamma", 0, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
        "reg_alpha":        trial.suggest_float("reg_alpha",  1e-8, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        # --- Reproducibility ---
        "random_state": param.random_seed,
        "n_jobs":       getattr(param, "n_jobs", N_JOBS),
        "verbosity":    0,
    }

    if is_regression:
        params["objective"]   = "reg:squarederror"
        params["eval_metric"] = "rmse"
    else:
        params["objective"]        = "binary:logistic"
        params["eval_metric"]      = "logloss"
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight",
            base_ratio * 0.5,
            base_ratio * 2.0,
        )

    return params

def build_logistic_search_space(
    param,
    trial: optuna.trial.Trial,
    base_ratio: float | None,
) -> dict:
    """
    Hyperparameter search space for logistic regression on a small dataset (~1100 samples).

    Mirror of :func:`build_hyperparameter_search_space`: fixed keys are set here, the
    tunable ones come from the trial, and the class-imbalance weight is scaled with the
    training-set negative/positive ratio, exactly like XGBoost's ``scale_pos_weight``.
    Only binary classification is supported: this project has no linear regression path.

    Parameters
    ----------
    param : argparse.Namespace
        General config. Uses param.random_seed for reproducibility.
    trial : optuna.trial.Trial
        Optuna trial object.
    base_ratio : float or None
        Ratio of negative to positive samples for classification.
        Pass None for regression tasks; that raises NotImplementedError.

    Returns
    -------
    dict
        Keyword arguments for ``LogisticRegression``, except ``scale_pos_weight``,
        which :meth:`Models.OurLinear.OurLinearModel._construct_model` converts into
        ``class_weight``.
    """
    if base_ratio is None:
        raise NotImplementedError(
            "Currently only the binary classification task is supported by logistic regression."
        )

    return {
        # --- Regularization ---
        "C":            trial.suggest_float("C", 1e-4, 1e2, log=True),
        "l1_ratio":     trial.suggest_float("l1_ratio", 0.0, 1.0),
        # --- Convergence control (saga is iterative, unlike liblinear) ---
        "max_iter":     trial.suggest_int("max_iter", 200, 2000),
        "tol":          trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        # --- Class imbalance ---
        # Named like the XGBoost knob it parallels and, unlike class_weight,
        # a plain float: _construct_model of OurLinear.py turns it into
        # class_weight={1: value}, so every value here is also a valid
        # LogisticRegression keyword.
        "scale_pos_weight": trial.suggest_float(
                                "scale_pos_weight",
                                base_ratio * 0.5,
                                base_ratio * 2.0,
                            ),
        # --- Reproducibility ---
        "random_state": param.random_seed,
        # 'saga' is the only solver accepting an elastic-net penalty, i.e. an
        # l1_ratio strictly between 0 (ridge) and 1 (lasso). n_jobs is left alone:
        # saga parallelises over classes, so with a binary target extra threads
        # buy nothing and only add oversubscription risk (see constant.N_JOBS).
        "solver":       "saga",
        # No 'penalty' key on purpose: it is deprecated since sklearn 1.8 and
        # removed in 1.10, where l1_ratio alone selects the ridge/lasso mix.
        # saga honours l1_ratio under the default penalty already, so the
        # searched l1_ratio is effective without naming a penalty.
    }
def build_lightgbm_search_space(
    param,
    trial: optuna.trial.Trial,
    base_ratio: float | None,
) -> dict:
    """
    Hyperparameter search space for LightGBM on a small dataset (~1100 samples).

    Mirror of :func:`build_hyperparameter_search_space`: fixed keys are set here,
    the tunable ones come from the trial, and the class-imbalance weight keeps the
    XGBoost name ``scale_pos_weight``, which LightGBM accepts natively too (unlike
    logistic regression, where OurLinear.py has to translate it into
    ``class_weight``).

    Parameters
    ----------
    param : argparse.Namespace
        General config. Uses param.random_seed for reproducibility.
    trial : optuna.trial.Trial
        Optuna trial object.
    base_ratio : float or None
        Ratio of negative to positive samples for classification.
        Pass None for the regression task.

    Returns
    -------
    dict
        Keyword arguments for ``LGBMClassifier`` or ``LGBMRegressor``.
    """
    params = {
        # --- Core tree structure ---
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        # LightGBM grows leaves best-first, so num_leaves and not max_depth is its
        # capacity knob; max_depth only caps the same tree from above.
        "num_leaves":        trial.suggest_int("num_leaves", 4, 64),
        "max_depth":         trial.suggest_int("max_depth", 2, 6),
        # --- Leaf regularisation ---
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 5.0),
        # --- Stochastic regularisation ---
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        # --- Explicit regularisation ---
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-8, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        # --- Reproducibility ---
        "random_state": param.random_seed,
        "n_jobs":       getattr(param, "n_jobs", N_JOBS),
        # 'subsample' is bagging, and bagging only runs every subsample_freq
        # iterations: under the library default of 0 the searched value would be
        # silently ignored, i.e. a dead knob.
        "subsample_freq": 1,
        # -1 (fatal only) rather than 0: 0 still prints the C++ library's
        # per-boosting-round "No further splits with positive gain" warning, of
        # which a fit on ~500 rows emits dozens. It is not a suggested key, so a
        # refit from study.best_params also ends up with it.
        "verbosity":    -1,
    }

    if base_ratio is None:
        params["objective"] = "regression"
        params["metric"]    = "rmse"
    else:
        params["objective"]        = "binary"
        params["metric"]           = "binary_logloss"
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight",
            base_ratio * 0.5,
            base_ratio * 2.0,
        )

    return params
