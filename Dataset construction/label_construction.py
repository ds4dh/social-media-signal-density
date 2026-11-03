"""Target label construction for the dispatch-centre series.

The label logic lives in target_construction.compute_labels, which serves both
target units and both target scopes from a single code path. This module only
re-exports it and keeps the separate ICU-based labelling used by the early
ICU experiments.
"""

import numpy as np
import pandas as pd

from constants import DATA_DIR
from target_construction import (  # noqa: F401  (re-exported for existing imports)
    LABELING_STRATEGIES,
    compute_labels,
    construct_labels,
    labelled_target_series,
    report_labels,
)


def construct_labels_and_add_icu_features(df, params):
    """
    Construct target labels and add lagged ICU features to the DataFrame.

    Strategies
    ----------
    "trends" : binary label = 1 if `current_icu` strictly increases from the
               current day to the next day, 0 otherwise.

    Added features (always computed alongside the label)
    ----------------------------------------------------
    icu_t           : current ICU count on day t
    icu_lag_1       : ICU count 1 day before
    icu_lag_3       : ICU count 3 days before
    icu_lag_7       : ICU count 7 days before
    icu_diff_1      : icu_t - icu_lag_1   (1-day change)
    icu_diff_3      : icu_t - icu_lag_3   (3-day change)
    icu_diff_7      : icu_t - icu_lag_7   (7-day change)
    icu_roll_mean_7 : 7-day rolling mean of ICU, ending on day t

    All features describe day t only — no future information.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame, must include a 'date' column.
    params : argparse.Namespace
        Must have a `labeling_strategy` attribute.

    Returns
    -------
    pd.DataFrame
        `df` with the new ICU feature columns and a 'label' column, sorted
        by date. Rows without a defined label are dropped.
    """
    

    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    # --- Load and prep the ICU series ---
    icu = pd.read_csv(DATA_DIR / "covid_over_time_GE.csv")
    icu['date'] = pd.to_datetime(icu['date'])
    icu = (icu[['date', 'current_icu']]
           .dropna(subset=['date'])
           .sort_values('date')
           .reset_index(drop=True))

    # --- Label: does ICU strictly increase from t to t+1? ---
    if params.labeling_strategy == "trends":
        icu['label'] = (icu['current_icu'].shift(-params.horizon) > icu['current_icu']).astype('Int64')
    elif params.labeling_strategy == "smooth_trends":
        icu['icu_smoothed'] = (
            icu['current_icu'].rolling(window=params.smoothing_window, min_periods=1).mean()
        )
        icu['label'] = (
            icu['icu_smoothed'].shift(-params.horizon) > icu['icu_smoothed']
        ).astype('Int64')
    else:
        raise ValueError(f"Labeling strategy '{params.labeling_strategy}' not recognized. " "Please choose a valid strategy.")

    # --- Autoregressive features (computed on the dense ICU series) ---
    icu['icu_t']        = icu['current_icu']
    icu['icu_lag_1']    = icu['current_icu'].shift(1)
    icu['icu_lag_3']    = icu['current_icu'].shift(3)
    icu['icu_lag_7']    = icu['current_icu'].shift(7)
    icu['icu_diff_1']   = icu['icu_t'] - icu['icu_lag_1']
    icu['icu_diff_3']   = icu['icu_t'] - icu['icu_lag_3']
    icu['icu_diff_7']   = icu['icu_t'] - icu['icu_lag_7']
    icu['icu_roll_mean_7'] = (icu['current_icu']
                              .rolling(window=7, min_periods=1)
                              .mean())

    # --- Merge labels + features into df ---
    feature_cols = ['icu_t', 'icu_lag_1', 'icu_lag_3', 'icu_lag_7',
                    'icu_diff_1', 'icu_diff_3', 'icu_diff_7',
                    'icu_roll_mean_7']
    df = df.merge(icu[['date', 'label'] + feature_cols], on='date', how='left')

    # Rows without a label can't be used for supervised training
    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    return df