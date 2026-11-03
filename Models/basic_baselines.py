from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
)
from scipy.stats import spearmanr
import warnings
import numpy as np
import pandas as pd


def compute_metrics_baselines(param, dataset: dict) -> dict:
    """
    Compute and return metrics for a deterministic baseline.

    Dispatches to regression or binary classification metrics based on
    param.labeling_strategy:
        - 'weekly_calls'    → regression metrics (RMSE, MAE, MAPE, R², Spearman)
        - 'weekly_trends'   → binary classification metrics (AUC, accuracy, ...)
        - 'surge_detection' → binary classification metrics (AUC, accuracy, ...)

    Baselines implemented
    ---------------------
    same_as_last       : predict next value as equal to the last observed value.
    moving_average     : predict next value as rolling mean of past `window` values.
    arima              : ARIMA one-step-ahead rolling forecast.
    persistence_surge  : predict surge = last observed binary label (lag-1).
    logistic_calls     : logistic regression on call volume features.
    majority_class     : always predict the majority class (no surge).

    Parameters
    ----------
    param : argparse.Namespace
        Must have:
            - approach              : str
            - labeling_strategy     : str
            - moving_average_window : int (default 4)
            - arima_order           : list[int] (default [1,1,1])
            - surge_threshold       : float (default 0.10)
    dataset : dict
        Dictionary with keys 'train', 'validation', 'test'.

    Returns
    -------
    dict
        Metrics dictionary appropriate to the task type.
    """

    prefix = 'calls' if param.main_focus == 'calls' else 'patients'
    approach   = param.approach
    is_binary  = getattr(param, 'labeling_strategy', 'weekly_calls') != 'weekly_calls'

    y_train = dataset['train']['label'].reset_index(drop=True)
    y_val   = dataset['validation']['label'].reset_index(drop=True)
    y_test  = dataset['test']['label'].reset_index(drop=True)

    # Full history available at test time (train + val)
    y_history = pd.concat([y_train, y_val], ignore_index=True)

    # Call volume feature columns (needed by logistic_calls)
    call_cols = [
        c for c in dataset['train'].columns
        if c.startswith(prefix)
    ]

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if approach == 'same_as_last':
        y_pred  = _same_as_last(y_history, y_test)
        y_proba = y_pred.astype(float) if is_binary else None

    elif approach == 'moving_average':
        window  = getattr(param, 'moving_average_window', 4)
        y_pred  = _moving_average(y_history, y_test, window=window)
        y_proba = y_pred.astype(float) if is_binary else None

    elif approach == 'arima':
        order           = tuple(getattr(param, 'arima_order', [1, 1, 1]))
        surge_threshold = getattr(param, 'surge_threshold', 0.20)
        horizon         = getattr(param, 'horizon', 1)

        if not is_binary:
            y_pred = _arima_forecast(y_history, y_test, order=order, horizon=horizon)
            return _regression_metrics(y_test, y_pred, approach=approach)

        # Binary: reconstruct raw call series from calls_lag_1d
        raw_train = dataset['train'][prefix + '_lag_1d'].shift(-1).dropna()
        raw_val   = dataset['validation'][prefix + '_lag_1d'].shift(-1).dropna()
        raw_test  = dataset['test'][prefix + '_lag_1d'].shift(-1).dropna()

        raw_history_tv = pd.concat([raw_train, raw_val], ignore_index=True)

        y_test_aligned = y_test.iloc[:len(raw_test)].reset_index(drop=True)
        y_val_aligned  = y_val.iloc[:len(raw_val)].reset_index(drop=True)

        y_pred, y_proba = _arima_surge_with_tuned_threshold(
            raw_history_tv=raw_history_tv, 
            raw_val=raw_val.reset_index(drop=True),
            raw_test=raw_test.reset_index(drop=True),
            y_val=y_val_aligned,
            y_test=y_test_aligned,
            surge_threshold=surge_threshold,
            order=order,
            horizon=horizon
            )

        return _binary_metrics(
            y_test_aligned, y_pred, y_proba, approach='arima_surge'
        )

    elif approach == 'persistence_surge':
        if not is_binary:
            raise ValueError(
                "persistence_surge is only valid for binary labeling strategies."
            )
        y_pred, y_proba = _persistence_surge(y_history, y_test)

    elif approach == 'logistic_calls':
        if not is_binary:
            raise ValueError(
                "logistic_calls is only valid for binary labeling strategies."
            )
        if not call_cols:
            raise ValueError(
                "No calls_ columns found in dataset. "
                "Ensure feature_set includes 'calls'."
            )
        y_pred, y_proba = _logistic_calls(
            dataset, call_cols, y_train, y_val, y_test,
            random_state=getattr(param, 'random_seed', 42),
        )

    elif approach == 'majority_class':
        if not is_binary:
            raise ValueError(
                "majority_class is only valid for binary labeling strategies."
            )
        majority = int(y_history.mode().iloc[0])
        y_pred   = np.full(len(y_test), fill_value=majority, dtype=int)
        y_proba  = y_pred.astype(float)
        print(f"[majority_class] Always predicting class {majority}.")

    else:
        raise ValueError(
            f"Unknown baseline approach: '{approach}'. "
            f"Choose from: same_as_last, moving_average, arima, "
            f"persistence_surge, logistic_calls, majority_class."
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    if is_binary:
        return _binary_metrics(y_test, y_pred, y_proba, approach=approach)
    else:
        return _regression_metrics(y_test, y_pred, approach=approach)


# ------------------------------------------------------------------
# Regression baselines
# ------------------------------------------------------------------

def _same_as_last(y_history: pd.Series, y_test: pd.Series) -> np.ndarray:
    """Naive persistence: predict last observed value."""
    full      = pd.concat([y_history, y_test], ignore_index=True)
    n_history = len(y_history)
    y_pred    = np.array([
        full.iloc[n_history + i - 1]
        for i in range(len(y_test))
    ])
    print(f"[same_as_last] {len(y_test)} test observations.")
    return y_pred


def _moving_average(
    y_history: pd.Series,
    y_test: pd.Series,
    window: int = 4,
) -> np.ndarray:
    """Rolling mean of past `window` observations."""
    full      = pd.concat([y_history, y_test], ignore_index=True)
    n_history = len(y_history)
    y_pred    = np.full(len(y_test), fill_value=np.nan)

    for i in range(len(y_test)):
        idx         = n_history + i
        window_vals = full.iloc[max(0, idx - window):idx].dropna()
        y_pred[i]   = window_vals.mean() if len(window_vals) > 0 else y_history.mean()

    print(f"[moving_average] window={window}, {len(y_test)} observations.")
    return y_pred


def _arima_forecast(
    y_history: pd.Series,
    y_test: pd.Series,
    order: tuple = (1, 1, 1),
    horizon: int=1,
) -> np.ndarray:
    """ARIMA rolling one-step-ahead forecast on raw call volume."""
    history = y_history.dropna().tolist()
    y_pred  = np.full(len(y_test), fill_value=np.nan)

    print(
        f"[arima] order={order}, horizon={horizon}, history={len(history)}, "
        f"forecasting {len(y_test)} steps."
    )

    for i, actual in enumerate(y_test):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fitted    = ARIMA(history, order=order).fit()
                forecast  = fitted.forecast(steps=horizon)
                # Handle both Series and ndarray return types
                y_pred[i] = float(forecast.iloc[0]) if hasattr(forecast, 'iloc') \
                            else float(forecast[0])
            except Exception as e:
                print(f"[arima] Step {i} failed ({e}), using last value.")
                y_pred[i] = history[-1]

        if not np.isnan(float(actual)):
            history.append(float(actual))

    return y_pred


# ------------------------------------------------------------------
# Binary classification baselines
# ------------------------------------------------------------------

def _persistence_surge(
    y_history: pd.Series,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict surge = last observed binary label (lag-1 persistence).
    If a surge happened last week, predict surge this week.
    """
    full      = pd.concat([y_history, y_test], ignore_index=True)
    n_history = len(y_history)
    y_pred    = np.array([
        int(full.iloc[n_history + i - 1])
        for i in range(len(y_test))
    ])
    y_proba = y_pred.astype(float)
    print(f"[persistence_surge] {len(y_test)} test observations.")
    return y_pred, y_proba


def _logistic_calls(
    dataset: dict,
    call_cols: list,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Logistic regression on call volume features.
    Trains on train+val combined, evaluates on test.
    """
    X_train = dataset['train'][call_cols].fillna(0)
    X_val   = dataset['validation'][call_cols].fillna(0)
    X_test  = dataset['test'][call_cols].fillna(0)

    # Train on train + val combined
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            C=0.1,
        ))
    ])
    clf.fit(X_tv, y_tv)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = clf.predict(X_test)

    print(
        f"[logistic_calls] Trained on {len(X_tv)} samples, "
        f"{len(call_cols)} features."
    )
    return y_pred, y_proba


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def _binary_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    approach: str = "",
) -> dict:
    """Compute and print binary classification metrics."""
    y_t = y_true.values.astype(int)

    try:
        auc = roc_auc_score(y_t, y_proba)
    except ValueError:
        auc = float('nan')

    acc  = accuracy_score(y_t, y_pred)
    prec = precision_score(y_t, y_pred, zero_division=0)
    rec  = recall_score(y_t, y_pred, zero_division=0)
    f1   = f1_score(y_t, y_pred, zero_division=0)

    print(
        f"\n[{approach}]"
        f"\n  AUC       : {auc:.4f}"
        f"\n  Accuracy  : {acc:.4f}"
        f"\n  Precision : {prec:.4f}"
        f"\n  Recall    : {rec:.4f}"
        f"\n  F1        : {f1:.4f}"
    )

    return {
        'roc_auc':   auc,
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
    }


def _regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    approach: str = "",
) -> dict:
    """Compute and print regression metrics."""
    mask = ~(np.isnan(y_pred) | y_true.isna())
    y_t  = y_true[mask].values
    y_p  = y_pred[mask]

    rmse            = np.sqrt(mean_squared_error(y_t, y_p))
    mae             = mean_absolute_error(y_t, y_p)
    r2              = r2_score(y_t, y_p)
    spearman, _     = spearmanr(y_t, y_p)
    nonzero         = y_t != 0
    mape            = (
        np.abs((y_t[nonzero] - y_p[nonzero]) / y_t[nonzero]).mean() * 100
        if nonzero.any() else np.nan
    )

    print(
        f"\n[{approach}]"
        f"\n  RMSE     : {rmse:.4f}"
        f"\n  MAE      : {mae:.4f}"
        f"\n  MAPE     : {mape:.2f}%"
        f"\n  R²       : {r2:.4f}"
        f"\n  Spearman : {spearman:.4f}"
    )

    return {
        "rmse": rmse, "mae": mae,
        "mape": mape, "r2": r2,
        "spearman": spearman,
    }


# Replace the fixed threshold binary conversion with a tuned threshold
# Find the threshold on validation that maximises F1

def _arima_surge_with_tuned_threshold(
    raw_history_tv: pd.Series,
    raw_val: pd.Series,
    raw_test: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    surge_threshold: float,
    order: tuple,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ARIMA surge detection with validation-tuned threshold.
    Uses the dataset labels directly rather than recomputing the
    surge threshold — this ensures consistency with the horizon
    used during label construction.
    """
    n_history_train   = len(raw_history_tv) - len(raw_val)
    raw_history_train = raw_history_tv.iloc[:n_history_train]

    # --- Validation forecasts ---
    val_forecasts = _arima_forecast(
        raw_history_train, raw_val, order=order, horizon=horizon
    )

    # Rolling 28-day baseline at each validation day
    full_tv   = raw_history_tv
    val_proba = np.full(len(raw_val), fill_value=np.nan)
    for i in range(len(raw_val)):
        idx         = n_history_train + i
        window_vals = full_tv.iloc[max(0, idx - 28):idx].dropna()
        roll_mean   = window_vals.mean() if len(window_vals) >= 14 \
                      else full_tv.iloc[:idx].mean()
        surge_level  = (1 + surge_threshold) * roll_mean
        val_proba[i] = val_forecasts[i] / surge_level if surge_level > 0 else 0.0

    # Tune threshold against the actual horizon=3 validation labels
    best_f1, best_thresh = 0.0, 0.5
    y_val_arr = y_val.iloc[:len(raw_val)].values.astype(int)
    for t in np.linspace(
        np.nanmin(val_proba), np.nanmax(val_proba), 200
    ):
        preds = (val_proba >= t).astype(int)
        f1    = f1_score(y_val_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(
        f"[arima_surge] horizon={horizon}, "
        f"Best val F1={best_f1:.4f} at threshold={best_thresh:.4f}"
    )

    # --- Test forecasts ---
    test_forecasts = _arima_forecast(
        raw_history_tv, raw_test, order=order, horizon=horizon
    )

    full_raw   = pd.concat([raw_history_tv, raw_test], ignore_index=True)
    n_tv       = len(raw_history_tv)
    test_proba = np.full(len(raw_test), fill_value=np.nan)
    for i in range(len(raw_test)):
        idx         = n_tv + i
        window_vals = full_raw.iloc[max(0, idx - 28):idx].dropna()
        roll_mean   = window_vals.mean() if len(window_vals) >= 14 \
                      else full_raw.iloc[:idx].mean()
        surge_level  = (1 + surge_threshold) * roll_mean
        test_proba[i] = test_forecasts[i] / surge_level if surge_level > 0 else 0.0

    y_pred = (test_proba >= best_thresh).astype(int)
    print(
        f"[arima_surge] predicted {y_pred.sum()} surges "
        f"out of {len(y_pred)} test days."
    )
    return y_pred, test_proba