"""Unique-tweet table, pool of informative tweets, and background corpus.

Task.md step 2. Decision 2: unique ``tweet_id``s everywhere. Decision 1: the LLM
labels are used *only* here, to decide which tweets are informative.

The pool is one single pool spanning the whole period: the injection draws every
day's tweets uniformly, with replacement, from all of it, so the same tweet can
land on a training day and on a test day of the same fold (``inject``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paths import (
    DISPATCH_UNIT,
    HORIZON,
    LABELING_STRATEGY,
    LLM_LABELS_PARAMS_JSON,
    LLM_LABELS_PARQUET,
    LLM_PROMPT_HASH_PREFIX,
    LOCAL_TZ,
    REFERENCE_DATASET_PKL,
    SURGE_THRESHOLD,
    TARGET_SCOPE,
    TWEETS_PKL,
)

#: tweets after this local date are outside the dispatch-centre ground truth
CUTOFF_LOCAL_DATE = pd.Timestamp("2023-01-01")

#: columns of the unique-tweet table
TWEET_COLUMNS = ["tweet_id", "day", "lang", "text"]

#: columns returned by :func:`load_llm_labels`
LLM_COLUMNS = ["tweet_id", "day", "p_yes", "informative", "status", "lang"]

#: columns of the injection pool (``day`` is the local posting date)
POOL_COLUMNS = ["tweet_id", "day", "lang"]


def load_unique_tweets(
    tweets_pkl=TWEETS_PKL,
    *,
    cutoff: pd.Timestamp = CUTOFF_LOCAL_DATE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return one row per unique tweet, ordered by ``tweet_id``.

    Columns: ``tweet_id, day`` (``YYYY-MM-DD``, Europe/Zurich), ``lang``, ``text``.

    Duplicate ``tweet_id``s (9,072 in the raw corpus) are collapsed keeping the
    first occurrence, then the frame is sorted by ``tweet_id`` so that every
    downstream order — cache rows, chunk boundaries — is independent of the
    order ``tweets.pkl`` happens to be in.
    """
    raw = pd.read_pickle(tweets_pkl)
    df = raw[["id", "created_at", "lang", "text"]].copy()
    df["tweet_id"] = df["id"].astype(str)
    df["day"] = (
        pd.to_datetime(df["created_at"], utc=True)
        .dt.tz_convert(LOCAL_TZ)
        .dt.strftime("%Y-%m-%d")
    )
    n_raw = len(df)
    df = df[df["day"] < cutoff.strftime("%Y-%m-%d")]
    n_cutoff = len(df)
    df = (
        df.drop_duplicates("tweet_id", keep="first")
        .sort_values("tweet_id", kind="mergesort")
        .reset_index(drop=True)
    )
    df["lang"] = df["lang"].fillna("und")
    if verbose:
        print(
            f"[load_unique_tweets] {n_raw} rows -> {n_cutoff} before "
            f"{cutoff.date()} -> {len(df)} unique tweet_ids."
        )
    return df[TWEET_COLUMNS].copy()


def load_llm_labels(labels_parquet=LLM_LABELS_PARQUET, *, verbose: bool = True) -> pd.DataFrame:
    """Return ``tweet_id, day, p_yes, informative, status, lang`` for the full run.

    ``day`` is ``date_local`` of the LLM labels, i.e. the posting date in
    ``Europe/Zurich`` — the same day the unique-tweet table derives from
    ``created_at``. It is kept on the pool for bookkeeping and reporting.
    """
    llm = pd.read_parquet(labels_parquet)
    llm["tweet_id"] = llm["tweet_id"].astype(str)
    missing = {"tweet_id", "date_local", "p_yes", "informative_hard", "status", "lang"} - set(llm.columns)
    if missing:
        raise KeyError(f"{labels_parquet} is missing columns: {sorted(missing)}")
    llm["day"] = pd.to_datetime(llm["date_local"]).dt.strftime("%Y-%m-%d")
    llm["informative"] = llm["informative_hard"].astype(bool)
    llm["status"] = llm["status"].astype(str)
    if verbose:
        print(
            f"[load_llm_labels] {len(llm)} scored tweets; "
            f"{int(llm.informative.sum())} informative; "
            f"statuses={llm.status.value_counts().to_dict()}"
        )
    return llm[LLM_COLUMNS].copy()


def check_prompt_hash(params_json=None, *, expected: str = LLM_PROMPT_HASH_PREFIX) -> str:
    """Return the prompt hash recorded for the LLM run, checking its prefix."""
    path = Path(params_json) if params_json is not None else LLM_LABELS_PARAMS_JSON
    recorded = json.loads(path.read_text())["prompt_hash"]
    if not recorded.startswith(expected):
        raise ValueError(
            f"LLM labels were produced with prompt hash {recorded[:12]}..., "
            f"expected {expected}... (Task.md step 1). Re-run the classification."
        )
    return recorded


def reference_days(reference_pkl=REFERENCE_DATASET_PKL) -> list[str]:
    """Return the reference-dataset days as ``YYYY-MM-DD`` strings, sorted.

    Decision: the experiment uses only the days present in the reference
    dataset, so that rows, folds and labels stay identical.
    """
    ds = pd.read_pickle(reference_pkl)
    days = sorted(pd.to_datetime(ds["date"]).dt.strftime("%Y-%m-%d"))
    span = pd.date_range(days[0], days[-1], freq="D").strftime("%Y-%m-%d")
    missing = sorted(set(span) - set(days))
    print(
        f"[reference_days] {len(days)} days ({days[0]} -> {days[-1]}); "
        f"{len(missing)} calendar day(s) in the span but absent from the "
        f"dataset: {missing}."
    )
    return days


def build_pool(llm: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Pool of informative tweets: ``status == 'ok'`` and ``informative == 1``.

    Columns ``tweet_id, day, lang``, where ``day`` is the local posting date.
    The injection draws every day uniformly from this one pool, so the same tweet
    can appear on both the training days and the test days of the same fold; that
    overlap is accepted by design. ``day`` is kept for bookkeeping and reporting.
    """
    pool = llm[(llm["status"] == "ok") & (llm["informative"])]
    pool = pool.drop_duplicates("tweet_id").sort_values("tweet_id", kind="mergesort")
    pool = pool.reset_index(drop=True)
    if verbose:
        mix = pool["lang"].value_counts()
        share = (100 * mix / max(len(pool), 1)).round(1).to_dict()
        print(f"[build_pool] pool size: {len(pool)} informative tweets (status 'ok').")
        print(f"[build_pool] language mix (count): {mix.to_dict()}")
        print(f"[build_pool] language mix (%):     {share}")
    return pool[POOL_COLUMNS].copy()


def build_background(
    tweets: pd.DataFrame,
    llm: pd.DataFrame,
    days: list[str],
    *,
    keep_real_informative: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Real non-informative tweets per day: the background the pool is added to.

    A tweet belongs to the background when it is ``informative == 0`` or its
    ``status`` is not ``'ok'``. Real informative tweets are removed so that the
    injected tweets are the only informative signal and lambda is controlled
    exactly; ``keep_real_informative=True`` keeps them (variant). Tweets the LLM
    never scored stay in the background: they carry no label, so they cannot
    leak the pool definition.
    """
    informative = set(llm.loc[(llm["status"] == "ok") & (llm["informative"]), "tweet_id"])
    if keep_real_informative:
        bg = tweets.copy()
    else:
        bg = tweets[~tweets["tweet_id"].isin(informative)].copy()
    n_unique = len(bg)
    bg = bg[bg["day"].isin(days)].reset_index(drop=True)
    if verbose:
        print(
            f"[build_background] keep_real_informative={keep_real_informative}: "
            f"{n_unique} unique tweets -> {len(bg)} on the {len(days)} reference "
            f"days ({len(bg) / max(len(days), 1):.1f} tweets/day)."
        )
    return bg[["tweet_id", "day", "lang"]].copy()


def dataset_params() -> dict:
    """Label configuration reproduced unchanged (decision 5)."""
    return {
        "labeling_strategy": LABELING_STRATEGY,
        "horizon": HORIZON,
        "surge_threshold": SURGE_THRESHOLD,
        "unit": DISPATCH_UNIT,
        "scope": TARGET_SCOPE,
        "reference_dataset": REFERENCE_DATASET_PKL.name,
    }
