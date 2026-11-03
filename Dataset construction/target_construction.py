"""Daily target counts and label construction for the dispatch-centre series.

One code path serves both target units ('calls' and 'patients') and both
target scopes ('total' and 'respiratory'): the unit and scope pick a dense
daily count series (dispatch_data.daily_target_series), and the labeling
strategy turns that series into a label. Adding a scope or a strategy
therefore never means duplicating a strategy block.

Label definitions (x = daily target counts, all windows inclusive):

'weekly_trend'    y_t = 1 if mean(x[t+1 : t+7]) > mean(x[t-6 : t]) else 0
    One-week-ahead trend direction, both windows 7 days.

'weekly_trends'   y_t = 1 if mean(x[t+7(h-1)+1 : t+7h]) > mean(x[t-6 : t]) else 0
    Same, but the future week shifts forward by one week per --horizon
    (the historical definition, kept for the earlier experiments).

'weekly_calls'    y_t = mean(x[t+7(h-1)+1 : t+7h])   (continuous)

'surge_detection' y_t = 1 if x[t+h] > (1 + surge_threshold) * mean(x[t-27 : t]) else 0
"""

import numpy as np
import pandas as pd

from dispatch_data import (  # noqa: F401  (TARGET_SCOPES re-exported for CLI choices)
    TARGET_SCOPES,
    daily_target_series,
    write_missing_pp_share,
)

LABELING_STRATEGIES = ("weekly_trend", "weekly_trends", "weekly_calls", "surge_detection")
BINARY_STRATEGIES = ("weekly_trend", "weekly_trends", "surge_detection")

TARGET_COL = "n_target"
PAST_WINDOW = 7
SURGE_BASELINE_WINDOW = 28
SURGE_MIN_PERIODS = 14


def labelled_target_series(params):
    """
    Return the dense daily target series with a 'label' column attached.

    Labels are computed on the full dense calendar series, never on the
    (possibly gapped) feature rows: a day without tweets must not stretch a
    7-day window into a 10-day window and silently change what the label means.

    Returns
    -------
    pd.DataFrame
        DataFrame(date, n_target[, missing_pp_share], label). 'label' is NaN
        where a window is incomplete.
    """
    strategy = params.labeling_strategy
    horizon = getattr(params, "horizon", 1)
    surge_threshold = getattr(params, "surge_threshold", 0.10)

    print(
        f"[construct_labels] strategy='{strategy}', "
        f"unit='{params.general_dispatch_center_focus}', "
        f"scope='{getattr(params, 'target_scope', 'total')}', horizon={horizon}"
    )

    target = daily_target_series(params).sort_values("date").reset_index(drop=True)
    target["date"] = pd.to_datetime(target["date"])

    if "missing_pp_share" in target.columns:
        write_missing_pp_share(target, getattr(params, "save_dir", None))

    target["label"] = compute_labels(
        target[TARGET_COL],
        target["date"],
        strategy,
        horizon=horizon,
        surge_threshold=surge_threshold,
    )
    report_labels(target["date"], target["label"].to_numpy(), strategy)

    if strategy in BINARY_STRATEGIES:
        target.loc[target["label"].notna(), "label"] = (
            target.loc[target["label"].notna(), "label"].astype(int)
        )
    return target


def construct_labels(df, params):
    """
    Build the target labels for every feature row in `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Feature rows with a 'date' column (one row per day).
    params : argparse.Namespace
        Uses 'general_dispatch_center_focus', 'target_scope',
        'labeling_strategy', 'horizon', 'surge_threshold'.

    Returns
    -------
    pd.DataFrame
        `df` with 'n_target' and 'label' columns, sorted by date. Rows outside
        the labelled calendar range are dropped. Binary strategies return
        integer labels.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    target = labelled_target_series(params)
    target = target.dropna(subset=["label"])

    cols = ["date", TARGET_COL, "label"]
    if "missing_pp_share" in target.columns:
        cols.append("missing_pp_share")

    before = len(out)
    out = out.merge(target[cols], on="date", how="inner")
    dropped = before - len(out)
    if dropped:
        print(
            f"[construct_labels] Dropped {dropped} day(s) without a defined target label "
            f"({out['date'].min().date()} → {out['date'].max().date()} kept)."
        )
    print(
        f"[construct_labels] {len(out)} day(s) joined with the target; "
        f"{int(out[TARGET_COL].sum())} target events in range."
    )
    if params.labeling_strategy in BINARY_STRATEGIES:
        out["label"] = out["label"].astype(int)
    return out.sort_values("date").reset_index(drop=True)


def _future_mean(series, horizon):
    """Rolling 7-day mean of days t+7(h-1)+1 … t+7h, aligned to day t."""
    return series.rolling(7).mean().shift(-7 * horizon)


def compute_labels(target_counts, dates, strategy, horizon=1, surge_threshold=0.10):
    """
    Compute the label series from a dense, date-sorted daily target count.

    Parameters
    ----------
    target_counts : array-like
        Daily counts on a dense calendar index (no missing days).
    dates : array-like
        The matching dates, used only to report the first labelled day.
    strategy : str
        One of LABELING_STRATEGIES.
    horizon : int
        Forecast horizon: weeks for the weekly_* strategies, days for
        'surge_detection'.
    surge_threshold : float
        Relative excess over the 28-day baseline for 'surge_detection'.

    Returns
    -------
    np.ndarray
        Label per day; NaN where the label is undefined (incomplete windows).
    """
    s = pd.Series(np.asarray(target_counts, dtype="float64"))

    if strategy == "weekly_trend":
        past_mean = s.rolling(PAST_WINDOW).mean()
        future_mean = _future_mean(s, 1)
        label = (future_mean > past_mean).astype("float64")
        label[past_mean.isna() | future_mean.isna()] = np.nan

    elif strategy == "weekly_trends":
        past_mean = s.rolling(PAST_WINDOW).mean()
        future_mean = _future_mean(s, horizon)
        label = (future_mean > past_mean).astype("float64")
        label[past_mean.isna() | future_mean.isna()] = np.nan

    elif strategy == "weekly_calls":
        label = _future_mean(s, horizon).to_numpy(dtype="float64")

    elif strategy == "surge_detection":
        baseline = s.rolling(SURGE_BASELINE_WINDOW, min_periods=SURGE_MIN_PERIODS).mean()
        future = s.shift(-horizon)
        label = (future > (1 + surge_threshold) * baseline).astype("float64")
        label[baseline.isna() | future.isna()] = np.nan

    else:
        raise ValueError(
            f"Unknown labeling strategy: '{strategy}'. Choose from: {list(LABELING_STRATEGIES)}."
        )

    label = np.asarray(label, dtype="float64")
    n_nan = int(np.isnan(label).sum())
    if n_nan:
        valid = pd.Series(list(dates))[~np.isnan(label)]
        first = valid.iloc[0].date() if len(valid) else "none"
        print(
            f"[compute_labels] {n_nan} day(s) have an incomplete window and get no label "
            f"(first labelled day: {first})."
        )
    return label


def report_labels(dates, labels, strategy):
    """Print label prevalence (binary) or distribution (continuous), plus per year."""
    valid = ~np.isnan(labels)
    n_total = int(valid.sum())
    if n_total == 0:
        print("[report_labels] No day has a fully defined label.")
        return

    if strategy not in BINARY_STRATEGIES:
        vals = labels[valid]
        print(
            f"[report_labels] Continuous target — mean={vals.mean():.1f}, "
            f"std={vals.std():.1f}, min={vals.min():.1f}, max={vals.max():.1f}, "
            f"valid days={n_total}"
        )
        return

    n_pos = int((labels[valid] == 1).sum())
    print(
        f"[report_labels] Label prevalence: {n_pos}/{n_total} positive days "
        f"({n_pos / n_total:.1%})"
    )
    years = pd.Series(pd.to_datetime(list(dates))[valid]).dt.year
    per_year = pd.DataFrame({"year": years, "label": labels[valid]}).groupby("year")["label"].agg(
        ["sum", "count"]
    )
    for year, row in per_year.iterrows():
        prevalence = row["sum"] / row["count"]
        print(f"  {year}: {int(row['sum'])}/{int(row['count'])} positive ({prevalence:.1%})")
