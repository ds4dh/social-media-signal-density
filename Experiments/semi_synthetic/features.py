r"""Daily feature construction from the per-tweet cache (Task.md step 4).

Aggregates a (possibly injected) tweet corpus per day with the same logic as
the real pipeline (decision 3), using the cached per-tweet features:

- ``lexicon``: per-entity daily counts, ``count_any``, ``n_tweets``
  (``construct_final_tweet_data`` count-mode semantics: a tweet counts once per
  entity it matches), with the row-based lag scheme of
  ``add_time_series_features``. Count mode produces no score/rate columns, so
  neither does this module (decision 3).
- ``bert``: daily mean of the cached per-tweet TwHIN-BERT CLS embeddings and
  ``n_tweets``, with the same row-based lags. When a ``reducer`` is given, the
  per-tweet embeddings are projected by a PCA fitted on the *full* embedding
  cache before the daily pooling, and the reduced columns keep the ``emb_``
  prefix; the per-fold PCA of ``Models/dataset.py`` must then be skipped
  (``args.skip_fold_pca``).

The *aggregation* mirrors the real pipeline; the temporal window does not: both
builders default to the last ``paths.FEATURE_WINDOW`` days (t, t-1, t-2) instead
of the real pipeline's 7 days. Pass ``lags=constants.NLP_FEATURE_LAGS`` to
restore the 7-day window.

The result is merged with the reference dataset's dispatch-centre history,
calendar features and labels so that rows, dates and labels stay identical to
the reference run.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from paths import (BERT_IDS, BERT_MATRIX, CACHE_DIR, FEATURE_WINDOW_LAGS,
                   LEXICON_CACHE, REFERENCE_DATASET_PKL)

#: Default lag set: the tweet features of day ``t`` plus ``FEATURE_WINDOW - 1``
#: row-based lags, i.e. the last ``FEATURE_WINDOW`` days. The real pipeline uses
#: the 7-day ``constants.NLP_FEATURE_LAGS = (1, 2, 3, 4, 5, 6)``; the experiment
#: narrows that window to the last 3 days (t, t-1, t-2), so the *aggregation*
#: still mirrors the real pipeline while the temporal context is shorter.
#: Pass ``lags=constants.NLP_FEATURE_LAGS`` to restore the real window.
DEFAULT_LAGS = FEATURE_WINDOW_LAGS

#: window of the optional rolling sum of count_any (--add_rolling_sum)
ROLLING_SUM_WINDOW = 7

#: column prefixes of every tweet-derived column in the reference dataset
_TWEET_PREFIXES = ("count_", "score_", "emb_", "llm_", "n_tweets")


# ---------------------------------------------------------------------------
# cache loading
# ---------------------------------------------------------------------------

def load_lexicon_indicators(cache_path=LEXICON_CACHE):
    """Return ``(tweet_ids, entity_names, indicator_matrix)`` from the cache."""
    frame = pd.read_parquet(cache_path)
    frame["tweet_id"] = frame["tweet_id"].astype(str)
    frame = frame.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)
    entities = [c for c in frame.columns if c != "tweet_id"]
    return (
        frame["tweet_id"].to_numpy(),
        entities,
        frame[entities].to_numpy().astype(np.int8),
    )


def load_bert_embeddings(matrix_path=BERT_MATRIX, ids_path=BERT_IDS):
    """Return ``(tweet_ids, embedding_matrix)`` from the cache."""
    matrix = np.load(matrix_path)
    ids = pd.read_parquet(ids_path)["tweet_id"].astype(str).to_numpy()
    order = np.argsort(ids, kind="mergesort")
    return ids[order], matrix[order]


def _row_of_day(days: list[str], corpus_days) -> np.ndarray:
    """Map each corpus row to its index in the chronological ``days`` list."""
    position = {d: i for i, d in enumerate(days)}
    rows = pd.Series(list(corpus_days)).map(position)
    missing = int(rows.isna().sum())
    if missing:
        raise ValueError(
            f"{missing} corpus tweet(s) fall on a day outside the {len(days)} "
            "reference days; restrict the corpus to those days first."
        )
    return rows.to_numpy(dtype="int64")


def _align_ids(corpus_ids, cache_ids, what: str) -> np.ndarray:
    """Row positions of ``corpus_ids`` inside the sorted ``cache_ids``."""
    indexer = pd.Index(cache_ids).get_indexer(pd.Index(corpus_ids))
    missing = int((indexer < 0).sum())
    if missing:
        raise ValueError(
            f"{missing} corpus tweet(s) have no {what} cache row; rebuild the cache."
        )
    return indexer


# ---------------------------------------------------------------------------
# BERT dimensionality reduction
# ---------------------------------------------------------------------------

def _cache_fingerprint(matrix_path, ids_path) -> str:
    """SHA-256 over the embedding matrix (incl. its npy header) and id table."""
    h = hashlib.sha256()
    with open(matrix_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    h.update(Path(ids_path).read_bytes())
    return h.hexdigest()[:16]


class BertReducer:
    """PCA fitted on the *full* per-tweet embedding cache.

    Deliberately independent of the train/val/test split: the basis and the
    component count are identical for every fold, seed and lambda, so the
    ``emb_*`` columns of an experiment share one meaning and the lambda sweep is
    internally comparable. Fitting on the full cache means test-window tweets
    contribute to the basis; no labels are involved, so this is unsupervised
    transduction rather than target leakage.

    ``transform`` keeps the ``emb_`` prefix on purpose: the reduced columns then
    flow through :func:`build_bert_daily_features`, ``FEATURE_GROUPS['bert']``
    and the lag construction unchanged, and only the daily *mean* of the
    projections ever reaches the model.
    """

    def __init__(self, pca: PCA, n_components: int, *, n_tweets: int,
                 variance: float, threshold: float, fingerprint: str):
        self.pca = pca
        self._k = int(n_components)
        self.n_tweets = int(n_tweets)
        self.variance = float(variance)
        self.threshold = float(threshold)
        self.fingerprint = fingerprint

    @property
    def n_components(self) -> int:
        return self._k

    def transform(self, emb: np.ndarray) -> np.ndarray:
        """Project per-tweet embeddings onto the first ``k`` fitted axes."""
        proj = self.pca.transform(emb)
        return proj[:, : self._k].astype(np.float32)

    def describe(self) -> dict:
        return {
            "fit_population": "full_bert_cache",
            "n_tweets": self.n_tweets,
            "n_features_in": int(self.pca.n_features_in_),
            "n_components": self.n_components,
            "variance_threshold": self.threshold,
            "variance_explained": round(self.variance, 6),
            "cache_fingerprint": self.fingerprint,
        }


def fit_bert_reducer(
    threshold: float,
    *,
    matrix_path=BERT_MATRIX,
    ids_path=BERT_IDS,
    cache_path=None,
    random_state: int | None = None,
    verbose: bool = True,
) -> BertReducer:
    """Fit PCA on every cached per-tweet embedding (all 41,714 of them).

    ``threshold`` is a cumulative explained-variance target in (0, 1). The fitted
    reducer is cached as ``cache/bert_pca_<fp>_<threshold>.pkl`` keyed by the
    embedding-cache fingerprint, so re-runs and resumed runs reuse one basis.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold!r}.")

    fingerprint = _cache_fingerprint(matrix_path, ids_path)
    path = Path(cache_path) if cache_path else (
        CACHE_DIR / f"bert_pca_{fingerprint}_{threshold:g}.pkl"
    )
    if path.exists():
        with open(path, "rb") as fh:
            reducer = pickle.load(fh)
        if getattr(reducer, "fingerprint", None) != fingerprint:
            # An explicit cache_path carries no fingerprint, so a cached basis
            # from a different embedding cache would be reused silently.
            if verbose:
                print(f"[bert_pca] {path.name} was fitted on a different embedding "
                      "cache; refitting.")
        else:
            if verbose:
                print(f"[bert_pca] reused {path.name}: k={reducer.n_components} "
                      f"variance={reducer.variance:.1%} on {reducer.n_tweets} tweets")
            return reducer

    cache_ids, matrix = load_bert_embeddings(matrix_path, ids_path)
    n_fit = min(len(cache_ids), matrix.shape[1])
    pca = PCA(n_components=n_fit, random_state=random_state, svd_solver="full")
    pca.fit(matrix)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = min(int(np.searchsorted(cumvar, threshold)) + 1, n_fit)
    reducer = BertReducer(
        pca, k, n_tweets=len(cache_ids), variance=cumvar[k - 1],
        threshold=threshold, fingerprint=fingerprint,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(reducer, fh)
    if verbose:
        print(f"[bert_pca] fitted k={k} of {matrix.shape[1]} dims explaining "
              f"{cumvar[k - 1]:.1%} variance on {len(cache_ids)} tweets "
              f"-> {path.name}")
    return reducer



# ---------------------------------------------------------------------------
# per-method daily features
# ---------------------------------------------------------------------------

def build_lexicon_daily_features(
    corpus: pd.DataFrame,
    days: list[str],
    *,
    cache_path=LEXICON_CACHE,
    lags=DEFAULT_LAGS,
    add_rolling_sum: bool = False,
) -> pd.DataFrame:
    """Daily lexicon counts for one (possibly injected) corpus.

    Mirrors ``construct_final_tweet_data`` in count mode: ``count_i`` counts the
    tweets of the day that match entity i at least once, ``count_any`` the tweets
    matching any entity, ``n_tweets`` all tweets of the day; days without tweets
    get 0. Count mode emits no rate/score columns, so neither does this function.
    """
    cache_ids, entities, matrix = load_lexicon_indicators(cache_path)
    rows = _row_of_day(days, corpus["day"])
    ind = matrix[_align_ids(corpus["tweet_id"].to_numpy(), cache_ids, "lexicon")]

    n_days = len(days)
    onehot = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (np.arange(len(rows)), rows)),
        shape=(len(rows), n_days),
    )
    counts = np.asarray(onehot.T @ ind).reshape(n_days, -1).astype(np.int64)  # (D, E)
    any_ind = ind.any(axis=1).astype(np.int64)
    count_any = np.bincount(rows, weights=any_ind, minlength=n_days).astype(np.int64)
    n_tweets = np.bincount(rows, minlength=n_days).astype(np.int64)

    base = pd.DataFrame({"date": pd.to_datetime(pd.Series(days).astype(str))})
    base["n_tweets"] = n_tweets
    for j, entity in enumerate(entities):
        base[f"count_{entity}"] = counts[:, j]
    base["count_any"] = count_any

    if add_rolling_sum:
        base["count_any_roll_sum_7d"] = (
            base["count_any"]
            .rolling(ROLLING_SUM_WINDOW, min_periods=1)
            .sum()
            .to_numpy()
        )

    return add_row_lags(base, lags)


def build_bert_daily_features(
    corpus: pd.DataFrame,
    days: list[str],
    *,
    matrix_path=BERT_MATRIX,
    ids_path=BERT_IDS,
    lags=DEFAULT_LAGS,
    reducer: BertReducer | None = None,
) -> pd.DataFrame:
    """Daily mean embedding for one (possibly injected) corpus.

    Mirrors ``daily_tweets_pooling`` with ``pooling_strategy='mean'``: days with
    no tweets get NaN embeddings (the real pipeline reindexes with NaN).

    ``reducer`` (a PCA fitted on the full embedding cache) projects each
    *tweet's* embedding before the daily averaging. Because PCA is linear,
    projecting then averaging equals averaging then projecting, so the daily
    column ``emb_j`` is the daily mean of the j-th reduced coordinate. The
    column count (and therefore the model's feature count) is then fixed at
    ``reducer.n_components`` for every lambda and seed.

    ``n_tweets`` is deliberately *not* emitted, so this path contributes only the
    content channel (mean embeddings and their lags). The experiment name
    ``*_without_n_tweets`` records that choice; the lexicon path still emits it.
    """
    cache_ids, matrix = load_bert_embeddings(matrix_path, ids_path)
    if reducer is not None:
        matrix = reducer.transform(matrix)
    rows = _row_of_day(days, corpus["day"])
    emb = matrix[_align_ids(corpus["tweet_id"].to_numpy(), cache_ids, "BERT")]

    n_days = len(days)
    onehot = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (np.arange(len(rows)), rows)),
        shape=(len(rows), n_days),
    )
    sums = np.asarray(onehot.T @ emb)  # (D, H)
    n_tweets = np.bincount(rows, minlength=n_days).astype(np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.divide(
            sums, n_tweets[:, None],
            out=np.full_like(sums, np.nan, dtype="float32"),
            where=n_tweets[:, None] > 0,
        )

    base = pd.DataFrame({"date": pd.to_datetime(pd.Series(days).astype(str))})
    # base["n_tweets"] = n_tweets
    for j in range(daily.shape[1]):
        base[f"emb_{j}"] = daily[:, j]

    return add_row_lags(base, lags)


def add_row_lags(df: pd.DataFrame, lags) -> pd.DataFrame:
    """Append ``<col>_lag<lag>d`` shifts for every tweet-feature column.

    Row-based (``DataFrame.shift`` on the chronologically sorted daily frame),
    exactly like ``final_dataset_construction.add_time_series_features``.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    cols = [c for c in df.columns if c != "date"]
    for lag in lags:
        for col in cols:
            df[f"{col}_lag{lag}d"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# merge with the reference dataset
# ---------------------------------------------------------------------------

def build_daily_features(
    corpus: pd.DataFrame,
    days: list[str],
    method: str,
    *,
    reference_pkl=REFERENCE_DATASET_PKL,
    **method_kwargs,
) -> pd.DataFrame:
    """Daily tweet features for ``corpus``, merged with the reference rows.

    The reference dataset contributes its non-tweet columns (dispatch-centre
    history, calendar features, ``n_target``, ``label``) on the same dates; its
    own tweet columns are dropped and replaced by the recomputed ones.
    ``method_kwargs`` are forwarded to the method's builder (e.g.
    ``cache_path``, ``add_rolling_sum``).
    """
    if method == "lexicon":
        feats = build_lexicon_daily_features(corpus, days, **method_kwargs)
    elif method == "bert":
        feats = build_bert_daily_features(corpus, days, **method_kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'lexicon', 'bert'.")

    ref = pd.read_pickle(reference_pkl)
    ref["date"] = pd.to_datetime(ref["date"])
    ref_tweet_cols = [c for c in ref.columns if c.startswith(_TWEET_PREFIXES)]
    ref = ref.drop(columns=ref_tweet_cols)

    out = ref.merge(feats, on="date", how="left")
    if len(out) != len(ref):
        raise AssertionError(
            f"Merging the daily features changed the row count "
            f"({len(ref)} -> {len(out)}); the day lists are inconsistent."
        )
    return out
