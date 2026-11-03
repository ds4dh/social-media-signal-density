
import pandas as pd
import numpy as np
from constant import DATASET_DIR, STRING_MATCHING_DATASET, BERT_DATASET, BASELINE_APPROACHES


def construct_dataset(args):
    """
    Construct the dataset according to the targetted approach
    """
    
    if args.approach in ('matching', *BASELINE_APPROACHES):
        df = pd.read_pickle(DATASET_DIR / f"dataset_{STRING_MATCHING_DATASET}.pkl")
        
    elif args.approach == 'BERT':
        df = pd.read_pickle(DATASET_DIR / f"dataset_{BERT_DATASET}.pkl")
    
    else:
        raise NotImplementedError('The approach {} is currently not implemented. Currently only matching or BERT is supported'.format(args.approach))
    
    # select only the relevant 
    df = select_features(df, args.feature_set)

    # cv datasets construction
    final_datasets = construct_cv_datasets(df, args=args)

    return final_datasets



def construct_cv_datasets(df, args, cv=3):
    """
    Construct chronologically ordered cross-validation folds for time-series
    data using an expanding training window and fixed validation and test sizes.

    By default, for BERT-based approaches, PCA is applied within each fold:
        - PCA is fit on the training set only (no leakage)
        - The fitted PCA is applied to validation and test sets
        - Number of components determined by args.pca_variance_threshold
          (default 0.85), capped to ensure samples/feature >= 3 in fold 1

    Set args.skip_fold_pca (truthy) to leave the embedding columns untouched.
    The semi-synthetic experiment does that: it reduces the per-tweet embeddings
    with a PCA fitted once on the full embedding cache before the daily pooling,
    so the basis and the component count are identical for every fold, seed and
    lambda. Reducing a second time per fold would compress already-reduced
    columns and tie the component count to the fold size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'date' column, already feature-selected.
    args : argparse.Namespace
        Must contain: dataset_split_strategy, test_set_proportion,
        validation_set_proportion, approach, pca_variance_threshold.
    cv : int
        Number of cross-validation folds. Default is 3.

    Returns
    -------
    list[dict]
        List of fold dicts in chronological order with keys
        'fold', 'train', 'validation', 'test'.
    """
    if args.dataset_split_strategy != "time_based":
        raise ValueError(
            f"construct_cv_datasets requires time_based split strategy, "
            f"got '{args.dataset_split_strategy}'."
        )

    df = (df
          .sort_values(by="date")
          .dropna(subset=["date"])
          .reset_index(drop=True))

    n         = len(df)
    test_size = int(n * args.test_set_proportion)
    val_size  = int(n * args.validation_set_proportion)

    if val_size != test_size:
        raise ValueError(
            f"This CV scheme requires val_size == test_size, "
            f"got val_size={val_size}, test_size={test_size}. "
            f"Set validation_set_proportion == test_set_proportion."
        )

    print(f"[construct_cv_datasets] Total samples : {n} days")
    print(f"[construct_cv_datasets] Val  size     : {val_size} days")
    print(f"[construct_cv_datasets] Test size     : {test_size} days")

    min_train_size = n - (cv + 1) * test_size
    if min_train_size <= 0:
        raise ValueError(
            f"Not enough data for {cv} folds. "
            f"Fold 1 would have {min_train_size} training days. "
            f"Reduce cv or the val/test proportions."
        )
    print(f"[construct_cv_datasets] Min train size (fold 1): {min_train_size} days")

    # Detect BERT approach — PCA only applies to BERT, and only when the caller
    # has not already reduced the embeddings itself (args.skip_fold_pca).
    is_bert = (getattr(args, 'approach', '') == 'BERT'
               and not getattr(args, 'skip_fold_pca', False))

    folds = []
    for k in range(cv):
        test_end   = n - (cv - 1 - k) * test_size
        test_start = test_end  - test_size
        val_end    = test_start
        val_start  = val_end   - val_size
        train_end  = val_start

        train_set = df.iloc[:train_end].copy()
        val_set   = df.iloc[val_start:val_end].copy()
        test_set  = df.iloc[test_start:test_end].copy()

        print(
            f"\n[Fold {k+1}/{cv}]"
            f"\n  train : {len(train_set):>4} days  "
            f"({df['date'].iloc[0]} → {df['date'].iloc[train_end - 1]})"
            f"\n  val   : {len(val_set):>4} days  "
            f"({df['date'].iloc[val_start]} → {df['date'].iloc[val_end - 1]})"
            f"\n  test  : {len(test_set):>4} days  "
            f"({df['date'].iloc[test_start]} → {df['date'].iloc[test_end - 1]})"
        )

        # Apply PCA within fold for BERT approach
        if is_bert:
            train_set, val_set, test_set = _apply_fold_pca(
                train_set, val_set, test_set,
                variance_threshold=getattr(
                    args, 'pca_variance_threshold', 0.85
                ),
                random_state=getattr(args, 'random_seed', None),
            )

        folds.append({
            "fold":       k + 1,
            "train":      train_set,
            "validation": val_set,
            "test":       test_set,
        })

    return folds

def _apply_fold_pca(
    train_set: pd.DataFrame,
    val_set: pd.DataFrame,
    test_set: pd.DataFrame,
    variance_threshold: float = 0.85,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA

    # Identify base and lag embedding columns
    emb_cols = sorted([
        c for c in train_set.columns
        if c.startswith('emb_') and 'lag' not in c
    ])
    lag_emb_cols = [
        c for c in train_set.columns
        if c.startswith('emb_') and 'lag' in c
    ]
    all_emb_cols = emb_cols + lag_emb_cols

    print(f"[_apply_fold_pca] Found {len(emb_cols)} base emb cols, "
          f"{len(lag_emb_cols)} lag emb cols.")
    if emb_cols:
        print(f"  First 3 base: {emb_cols[:3]}")
    if lag_emb_cols:
        print(f"  First 3 lag:  {lag_emb_cols[:3]}")

    if not emb_cols:
        print("  [WARNING] No embedding columns found — PCA skipped.")
        return train_set, val_set, test_set

    # --- Step 1: Fit PCA on training data only ---
    X_train = train_set[emb_cols].values
    valid   = ~np.isnan(X_train).any(axis=1)
    n_train = valid.sum()

    # Cap components to maintain samples/feature >= 3
    # accounting for 4 lag versions per component + 18 call/time features
    max_components = max(1, int((n_train / 3 - 18) / 4))
    n_fit    = min(max_components, n_train - 1, len(emb_cols))
    # random_state must be pinned: at these shapes (431-751 rows x 768
    # emb columns) sklearn's 'auto' rule picks svd_solver='randomized',
    # whose randomized SVD depends on the global RNG. Unseeded, two
    # identical runs get different components and the fold AUROC moves
    # by up to ~0.04 — more than the gains being measured.
    pca_full = PCA(n_components=n_fit, random_state=random_state)
    pca_full.fit(X_train[valid])

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k      = int(np.searchsorted(cumvar, variance_threshold)) + 1
    k      = min(k, max_components, n_fit)
    k      = max(k, 1)

    print(
        f"[_apply_fold_pca] threshold={variance_threshold:.0%} → "
        f"k={k} components, explaining {cumvar[k-1]:.1%} of variance "
        f"(n_train={n_train}, samples/feature="
        f"{n_train / (k * 4 + 18):.1f})."
    )

    pca_cols = [f'pca_{i}' for i in range(k)]

    # --- Step 2: Concatenate all splits chronologically ---
    # Lags must be computed on the full series to avoid boundary NaNs
    n_train_rows = len(train_set)
    n_val_rows   = len(val_set)

    full_df = pd.concat(
        [train_set, val_set, test_set], axis=0
    ).reset_index(drop=True)

    # --- Step 3: Transform all splits with the fitted PCA ---
    X_full = full_df[emb_cols].values
    mask   = ~np.isnan(X_full).any(axis=1)

    X_pca_full        = np.full((len(X_full), k), np.nan)
    X_pca_full[mask]  = pca_full.transform(X_full[mask])[:, :k]

    pca_df = pd.DataFrame(
        X_pca_full,
        columns=pca_cols,
        index=full_df.index,
    )

    # --- Step 4: Compute lag features on full chronological series ---
    # Doing this before re-splitting ensures boundary lag values are correct
    for lag in [1, 3, 7]:
        for col in pca_cols:
            pca_df[f'{col}_lag{lag}d'] = pca_df[col].shift(lag)

    # --- Step 5: Drop original embedding columns, attach PCA + lags ---
    full_df = full_df.drop(columns=all_emb_cols, errors='ignore')
    full_df = pd.concat([full_df, pca_df], axis=1)

    # --- Step 6: Re-split at original boundaries ---
    train_out = full_df.iloc[:n_train_rows].copy()
    val_out   = full_df.iloc[n_train_rows:n_train_rows + n_val_rows].copy()
    test_out  = full_df.iloc[n_train_rows + n_val_rows:].copy()

    return train_out, val_out, test_out


def split_dataset(df, args):
    """
    Split the dataset into train, validation and test sets according to the specified strategy and proportions.
    """
    # check that the specified proportions sum to 1
    total_proportion = args.train_set_proportion + args.validation_set_proportion + args.test_set_proportion
    if total_proportion != 1.0:
        raise ValueError(f"The specified proportions for train, validation and test sets must sum to 1. Currently, they sum to {total_proportion}.")
    
    if args.dataset_split_strategy == "time_based":

        
        # sort the dataset by date
        df = df.sort_values(by="date")

        df.dropna(subset=["date"], inplace=True)

        # compute the number of samples for each set
        total_samples = len(df)
        train_samples = int(total_samples * args.train_set_proportion)
        validation_samples = int(total_samples * args.validation_set_proportion)
        test_samples = int(total_samples * args.test_set_proportion)

        # split the dataset
        train_set = df.iloc[:train_samples]
        validation_set = df.iloc[train_samples:train_samples + validation_samples]
        test_set = df.iloc[train_samples + validation_samples:train_samples + validation_samples + test_samples]

    return {
        "train": train_set,
        "validation": validation_set,
        "test": test_set
    }


# ---------------------------------------------------------------------------
# Feature groups used by --feature_set
#
# Column naming conventions produced by `Dataset construction/`:
#   {unit}_lag_{1,3,7}d, {unit}_roll_mean_7d  dispatch-centre history; the unit
#                                            is 'calls' or 'patients'
#                                            (dispatch_center_features.py)
#   icu_*                                    legacy ICU features
#                                            (label_construction.py)
#   time_*                                   calendar features
#                                            (add_calendar_features)
#   google_trends_<region>_<term>            Google Trends
#                                            (final_dataset_construction)
#   score_<entity>                           lexicon rate mode  (--aggregation rate)
#   count_<entity>, count_any                lexicon count mode (--aggregation count)
#   emb_<i>                                  BERT embedding dimensions
#   llm_count_*                              LLM relevance counts (todo D)
#   n_tweets                                 daily tweet volume, no group prefix
#
# Every NLP column is also lagged as '<column>_lag{1,3,7}d'
# (final_dataset_construction.add_time_series_features), so a prefix matches the
# base column and its lags at once.
# ---------------------------------------------------------------------------

#: prefix of the daily tweet-volume column ('n_tweets' and its lags)
N_TWEETS_PREFIX = "n_tweets"

# feature-group keyword -> the column prefixes that define the group
FEATURE_GROUPS = {
    "calls":         ("calls_",),
    "patients":      ("patients_",),
    "icu":           ("icu_",),
    "time":          ("time_",),
    "google_trends": ("google_trends_",),
    "tweets":        ("score_", "count_", "emb_", "llm_"),
    "matching":      ("score_", "count_"),
    "bert":          ("emb_",),
    "llm":           ("llm_",),
    "tweet_volume":  (N_TWEETS_PREFIX,),
}

#: NLP groups that also get the tweet-volume column with them
N_TWEETS_GROUPS = frozenset({"tweets", "matching", "bert", "llm"})

#: always kept: the time index and the target
METADATA_COLUMNS = ("date", "label")

#: never model inputs: 'n_target' is the target series itself at day t, and
#: 'missing_pp_share' is a reporting-bias diagnostic saved alongside the dataset.
NON_FEATURE_COLUMNS = ("n_target", "missing_pp_share")


def _in_group(col, group):
    """True if `col` is one of the columns that define `group`."""
    return col.startswith(FEATURE_GROUPS[group])


def _in_group_with_volume(col, group):
    """Like `_in_group`, plus the tweet-volume column for the NLP groups."""
    return _in_group(col, group) or (
        col.startswith(N_TWEETS_PREFIX) and group in N_TWEETS_GROUPS
    )


def select_features(df, feature_set):
    """
    Keep only the columns belonging to the requested feature groups.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame as produced by `Dataset construction/`.
    feature_set : str or list[str]
        Feature-group keywords to keep; valid keywords are the keys of
        ``FEATURE_GROUPS``:

            - 'calls' / 'patients' : dispatch-centre history of that unit
              (lags 1/3/7d plus the 7-day rolling mean)
            - 'icu'                : legacy ICU features
            - 'time'               : calendar features (day-of-week, weekend)
            - 'google_trends'      : Google Trends series
            - 'tweets'             : every tweet-derived column (score_*,
              count_*, emb_*, llm_*, n_tweets)
            - 'matching'           : lexicon columns only (score_*, count_*,
              count_any, n_tweets)
            - 'bert'               : embedding columns only (emb_*, n_tweets)
            - 'llm'                : LLM relevance columns only (llm_*, n_tweets)
            - 'tweet_volume'       : n_tweets only

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date', 'label' and the selected feature columns, in
        their original order.

    Raises
    ------
    ValueError
        If `feature_set` is empty, contains an unknown keyword, or selects a
        group that matches no column of `df` (which would silently train the
        model without that feature family).
    """
    if isinstance(feature_set, str):
        feature_set = [feature_set]
    feature_set = list(feature_set)

    if not feature_set:
        raise ValueError(
            f"feature_set is empty. Valid options: {sorted(FEATURE_GROUPS)}"
        )

    unknown = sorted(set(feature_set) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(
            f"Unknown feature group(s): {unknown}. "
            f"Valid options: {sorted(FEATURE_GROUPS)}"
        )

    candidate_cols = [
        c for c in df.columns
        if c not in METADATA_COLUMNS and c not in NON_FEATURE_COLUMNS
    ]

    # Every requested group must find at least one of its own columns: an empty
    # group means the dataset and the modelling code disagree on terminology,
    # and training on whatever is left would hide that.
    empty_groups = [g for g in feature_set
                    if not any(_in_group(c, g) for c in candidate_cols)]
    if empty_groups:
        families = sorted({c.split("_")[0] for c in candidate_cols})
        raise ValueError(
            f"Feature group(s) {empty_groups} match no column of the dataset "
            f"(shape={df.shape}). Column families present: {families}. "
            f"Requested prefixes: "
            f"{ {g: FEATURE_GROUPS[g] for g in feature_set} }"
        )

    # Columns that no group claims are invisible to every ablation: warn rather
    # than drop them silently.
    unclaimed = [
        c for c in candidate_cols
        if not any(_in_group_with_volume(c, g) for g in FEATURE_GROUPS)
    ]
    if unclaimed:
        print(f"[select_features] [WARNING] {len(unclaimed)} column(s) belong to "
              f"no known feature group and were dropped: {unclaimed}")

    columns_to_keep = [
        c for c in df.columns
        if c in METADATA_COLUMNS
        or (c not in NON_FEATURE_COLUMNS
            and any(_in_group_with_volume(c, g) for g in feature_set))
    ]
    per_group = {g: sum(_in_group_with_volume(c, g) for c in candidate_cols)
                 for g in feature_set}
    print(f"[select_features] Keeping {len(columns_to_keep)} of {df.shape[1]} "
          f"columns (groups: {per_group})")

    return df[columns_to_keep].copy()
