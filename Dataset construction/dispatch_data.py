"""Daily count series built from the two dispatch-centre exports.

'appels.csv' holds one row per emergency call. 'M_final.csv' holds one row
per emergency case (patient folio) and is the only export that carries the
main problem code ('p.PP'). Both files store naive timestamps in local Swiss
time, so '.dt.date' already yields local dates — the same convention used for
the tweets (see final_dataset_construction.tweet_local_dates).

The raw files are never modified here: every correction applied to the counts
happens in memory and is documented next to the code that applies it.
"""

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from constants import BASE_DIR, DATA_DIR, SHORT_LIST

# Main problem codes for respiratory / influenza-like presentations:
# 1008 (dyspnoea, breathing difficulty) and 1502 (fever, influenza-like state).
RESPIRATORY_PP_CODES = tuple(SHORT_LIST)

# Export artefact: the call count on this day is about 5x the neighbouring
# days (3487 calls against ~560), which no patient-side series confirms.
ANOMALY_DATE = pd.Timestamp("2021-01-04")

TARGET_SCOPES = ("total", "respiratory")

_PP_CODE_RANGE = re.compile(r"(\d{4})\s*-\s*(\d{4})")
_PP_CODE = re.compile(r"\b(\d{4})\b")


def pp_matches_codes(value, codes=RESPIRATORY_PP_CODES):
    """
    Return True if a main-problem string carries one of `codes`.

    The code sits in brackets at the end of a free-text label, sometimes as a
    range ('Fièvre (y.c. état grippal) (1501-1502)'), and the surrounding text
    is inconsistent across export years ('Dyspnée (1008)', 'Dyspnée (1008))').
    A plain substring test would miss ranges and could match unrelated numbers
    inside the text, so the codes are extracted before comparing.
    """
    if not isinstance(value, str):
        return False

    codes_as_int = {int(code) for code in codes}
    for low, high in _PP_CODE_RANGE.findall(value):
        if any(int(low) <= code <= int(high) for code in codes_as_int):
            return True

    return codes_as_int & {int(code) for code in _PP_CODE.findall(value)} != set()


def _dense_daily(dates, column):
    """Count rows per calendar day, then fill gap days with 0."""
    counts = dates.value_counts().sort_index()
    full_range = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="D")
    counts = counts.reindex(full_range, fill_value=0)
    return pd.DataFrame({"date": full_range, column: counts.to_numpy()})


@lru_cache(maxsize=None)
def _load_call_dates():
    """Return the local call timestamps from appels.csv as a Series."""
    appels = pd.read_csv(DATA_DIR / "appels.csv", usecols=["DateAppel"], low_memory=False)
    return pd.to_datetime(appels["DateAppel"], format="mixed", errors="coerce").dropna()


@lru_cache(maxsize=None)
def _load_cases():
    """
    Return the case export reduced to the columns the pipeline needs:
    DataFrame(date, is_respiratory, is_missing_pp).
    """
    cases = pd.read_csv(
        DATA_DIR / "M_final.csv",
        usecols=["p.PP", "f.DateOuverture"],
        low_memory=False,
    )
    cases["date"] = pd.to_datetime(cases["f.DateOuverture"], errors="coerce").dt.normalize()
    n_dropped = int(cases["date"].isna().sum())
    if n_dropped:
        print(f"[load_cases] Dropped {n_dropped} case(s) without a usable opening date.")
        cases = cases.dropna(subset=["date"])

    cases = cases.assign(
        is_respiratory=cases["p.PP"].map(pp_matches_codes).astype(int),
        is_missing_pp=cases["p.PP"].isna().astype(int),
    )
    return cases[["date", "is_respiratory", "is_missing_pp"]]


@lru_cache(maxsize=None)
def _load_daily_calls():
    """
    Return a dense daily call series: DataFrame(date, n_calls).

    The artefactual spike on ANOMALY_DATE is replaced by the mean of the two
    preceding days. Averaging the neighbouring days — the previous behaviour —
    pushed the call count of day t+1 into the value stored at day t, so every
    feature and label anchored at day t saw calls from day t+1.
    """
    daily = _dense_daily(_load_call_dates().dt.normalize(), "n_calls").set_index("date")

    if ANOMALY_DATE in daily.index:
        daily["n_calls"] = daily["n_calls"].astype("float64")
        trailing_mean = 0.5 * (
            daily["n_calls"].shift(1) + daily["n_calls"].shift(2)
        ).loc[ANOMALY_DATE]
        print(
            f"[load_daily_calls] Anomalous call count on {ANOMALY_DATE.date()} "
            f"({int(daily.loc[ANOMALY_DATE, 'n_calls'])}) replaced by the trailing "
            f"two-day mean ({trailing_mean:.1f})."
        )
        daily.loc[ANOMALY_DATE, "n_calls"] = trailing_mean

    return daily.reset_index()


@lru_cache(maxsize=None)
def _load_daily_cases():
    """
    Return a dense daily case series from the patient export:

        DataFrame(date, n_cases, n_respiratory, n_missing_pp, missing_pp_share)

    n_respiratory counts cases whose main problem code (p.PP) is in
    RESPIRATORY_PP_CODES, and missing_pp_share is the daily share of cases with
    no p.PP value (a reporting-bias check on the respiratory target).
    """
    cases = _load_cases()
    daily = (
        cases.groupby("date")
        .agg(
            n_cases=("is_respiratory", "size"),
            n_respiratory=("is_respiratory", "sum"),
            n_missing_pp=("is_missing_pp", "sum"),
        )
        .sort_index()
    )
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0).rename_axis("date").reset_index()

    daily["missing_pp_share"] = (daily["n_missing_pp"] / daily["n_cases"]).where(
        daily["n_cases"] > 0, 0.0
    )
    return daily


def load_daily_calls():
    """Return a copy of the dense daily call series (see _load_daily_calls)."""
    return _load_daily_calls().copy()


def load_daily_cases():
    """Return a copy of the dense daily case series (see _load_daily_cases)."""
    return _load_daily_cases().copy()


def load_dispatch_event_dates(focus):
    """
    Return the set of calendar dates that carry at least one dispatch record.

    Unlike the dense target series, gap days inside the coverage span are not
    reported as present, so the daily index built from these dates only keeps
    days with real dispatch-centre coverage.

    Parameters
    ----------
    focus : str
        'calls' (appels.csv) or 'patients' (M_final.csv).
    """
    if focus == "calls":
        dates = set(_load_call_dates().dt.date)
    elif focus == "patients":
        dates = set(_load_cases()["date"].dt.date)
    else:
        raise NotImplementedError(
            "The parameter --general_dispatch_center_focus can take the values 'calls' or 'patients'."
        )
    return dates


def daily_target_series(params):
    """
    Return the dense daily target counts for the configured target unit and scope.

    Parameters
    ----------
    params : argparse.Namespace
        Uses 'general_dispatch_center_focus' ('calls' or 'patients') as the
        target unit, and 'target_scope' ('total' or 'respiratory', default
        'total') as the case selection.

    Returns
    -------
    pd.DataFrame
        DataFrame(date, n_target[, missing_pp_share]). 'missing_pp_share' is
        present only when the target comes from the case export, which is the
        only export carrying p.PP.
    """
    focus = params.general_dispatch_center_focus
    scope = getattr(params, "target_scope", "total")

    if scope not in TARGET_SCOPES:
        raise ValueError(
            f"The parameter --target_scope can take the values {list(TARGET_SCOPES)}, got {scope!r}."
        )
    if focus not in ("calls", "patients"):
        raise NotImplementedError(
            "The parameter --general_dispatch_center_focus can take the values 'calls' or 'patients'."
        )

    if scope == "respiratory":
        # p.PP exists only in the case export, so a respiratory target is always
        # counted there: one case row is one emergency call-out carrying a main
        # problem code. Focus then only decides which unit the features describe.
        source, column = _load_daily_cases(), "n_respiratory"
        unit = "respiratory cases (one row per case with a respiratory main problem code)"
    elif focus == "calls":
        source, column = _load_daily_calls(), "n_calls"
        unit = "calls (one row per call record in appels.csv)"
    else:
        source, column = _load_daily_cases(), "n_cases"
        unit = "patients (one row per case in M_final.csv)"

    target = pd.DataFrame({"date": source["date"], "n_target": source[column]})
    if "missing_pp_share" in source.columns:
        target["missing_pp_share"] = source["missing_pp_share"]

    print(
        f"[daily_target_series] unit={focus}, scope={scope} → {unit}.\n"
        f"[daily_target_series] {int(target['n_target'].sum())} events over "
        f"{len(target)} days ({target['date'].min().date()} → {target['date'].max().date()}), "
        f"per day: mean={target['n_target'].mean():.1f}, median={target['n_target'].median():.0f}, "
        f"min={target['n_target'].min()}, max={target['n_target'].max()}."
    )
    if "missing_pp_share" in target.columns:
        print(
            f"[daily_target_series] Daily share of cases with a missing p.PP: "
            f"mean={target['missing_pp_share'].mean():.3f}, "
            f"min={target['missing_pp_share'].min():.3f}, "
            f"max={target['missing_pp_share'].max():.3f}."
        )

    return target


def write_missing_pp_share(target, output_dir=None):
    """
    Save the daily share of cases with a missing main problem code.

    Written as 'missing_pp_share.csv' with two columns (date, share) so the
    respiratory target can be checked against p.PP reporting bias. Days without
    any case get share 0.0. Does nothing when the target does not come from the
    case export.
    """
    if "missing_pp_share" not in target.columns:
        print(
            "[write_missing_pp_share] Skipped: the call export carries no p.PP field, "
            "so the reporting-bias series is undefined for this target."
        )
        return None

    output_dir = Path(output_dir) if output_dir is not None else BASE_DIR / "FINAL_DATASET"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "missing_pp_share.csv"
    target[["date", "missing_pp_share"]].rename(
        columns={"missing_pp_share": "share"}
    ).to_csv(csv_path, index=False)

    print(f"[write_missing_pp_share] Daily p.PP missingness saved to {csv_path} ({len(target)} rows).")
    return csv_path


