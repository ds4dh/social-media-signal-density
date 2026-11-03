import numpy as np
import pandas as pd

from constants import FEATURE_LAGS
from dispatch_data import daily_target_series, load_dispatch_event_dates

# Trailing window over the observed history, ending on day t.
ROLL_WINDOW = 7


def get_dispatch_dates(focus):
    """
    Return the set of calendar dates present in the dispatch-centre data.

    The dispatch CSVs store naive timestamps in local Swiss time, so taking
    '.dt.date' already yields local dates — the same convention used for the
    tweets (see final_dataset_construction.tweet_local_dates).

    Parameters
    ----------
    focus : str
        'calls' (appels.csv) or 'patients' (M_final.csv).
    """
    days = load_dispatch_event_dates(focus)
    print(
        f"[get_dispatch_dates] focus='{focus}': {len(days)} distinct days "
        f"({min(days)} → {max(days)})."
    )
    return days


def add_dispatch_center_features(df, params):
    """
    Add causal history features from the dispatch-centre target series.

    The series follows the same unit ('--general_dispatch_center_focus') and
    scope ('--target_scope') as the label, so features and target describe the
    same quantity: total calls, total patients, or respiratory cases. Feature
    names always start with the unit name ('calls_' or 'patients_'), which is
    how the modelling code selects its feature group.

    Features added (x = daily target counts)
    ---------------------------------------
    {unit}_lag_1d, _lag_3d, _lag_7d : x at t-1, t-3, t-7 — the same lags as the
        tweet features (final_dataset_construction.add_time_series_features).
    {unit}_roll_mean_7d : mean of x[t-6 : t], both ends included. This is the
        same window the 'weekly_trend' label compares against.

    Every feature at day t uses days t and earlier only, so no feature sees
    calls after t.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame with a 'date' column (one row per day).
    params : argparse.Namespace
        Uses 'general_dispatch_center_focus', 'target_scope', and optionally
        'dispatch_feature_lags' (defaults to the tweet lags).

    Returns
    -------
    pd.DataFrame
        df with the history features appended, sorted by date.
    """
    unit = params.general_dispatch_center_focus
    scope = getattr(params, "target_scope", "total")
    lags = list(getattr(params, "dispatch_feature_lags", FEATURE_LAGS))

    print("#" * 70)
    print(f"Adding daily {unit} history features (scope='{scope}', lags={lags}).")
    print("#" * 70)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    daily = daily_target_series(params)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    x = daily["n_target"]

    # --- Lag features: shift(k) puts day t-k at day t (strictly causal) ---
    for lag in lags:
        daily[f"{unit}_lag_{lag}d"] = x.shift(lag)

    # --- Rolling mean over the 7 days ending on day t (x[t-6 : t]) ---
    daily[f"{unit}_roll_mean_{ROLL_WINDOW}d"] = x.rolling(
        window=ROLL_WINDOW, min_periods=ROLL_WINDOW
    ).mean()

    feature_cols = [c for c in daily.columns if c.startswith(f"{unit}_")]
    df = df.merge(daily[["date"] + feature_cols], on="date", how="left")

    n_missing = int(df[feature_cols].isna().any(axis=1).sum())
    print(
        f"[add_dispatch_center_features] Added {len(feature_cols)} daily {unit} features "
        f"(scope='{scope}'): lags {lags}, roll_mean_{ROLL_WINDOW}d."
    )
    if n_missing:
        print(
            f"[add_dispatch_center_features] {n_missing} day(s) have incomplete {unit} "
            f"history (start of the series)."
        )
    return df.sort_values("date").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclically encoded calendar features for day-of-week and month.

    Cyclical encoding using sine and cosine transforms preserves the
    circular structure of calendar variables — ensuring that Monday
    (dow=0) and Sunday (dow=6) are recognised as adjacent, and that
    December and January are recognised as adjacent.

    Features added
    --------------
    time_day_sin, time_day_cos   : cyclical encoding of day of week (period=7)
    time_is_weekend         : binary flag for Saturday/Sunday

    Parameters
    ----------
    df : pd.DataFrame
        Must include a 'date' column parseable as datetime.

    Returns
    -------
    pd.DataFrame
        df with calendar features appended.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    dow   = df['date'].dt.dayofweek   # 0=Monday, 6=Sunday

    # Cyclical encoding — preserves circular adjacency
    df['time_day_sin']   = np.sin(2 * np.pi * dow   / 7)
    df['time_day_cos']   = np.cos(2 * np.pi * dow   / 7)

    # Binary flags — interpretable and directly actionable
    df['time_is_weekend'] = (dow >= 5).astype(int)

    print(
        f"[add_calendar_features] Added 3 calendar features: "
        f"time_day_sin, time_day_cos, time_is_weekend."
    )
    return df