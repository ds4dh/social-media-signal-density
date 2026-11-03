
import numpy as np
import pandas as pd
import torch
from constants import *
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from textual_features import construct_textual_features


LOCAL_TZ = "Europe/Zurich"


def tweet_local_dates(series):
    """
    Return the local (Europe/Zurich) calendar dates of tweet timestamps.

    Tweet 'created_at' values are UTC, while the dispatch-centre timestamps
    are naive local Swiss times. Converting tweets to local time before
    taking the date makes both sources bucketed on the same calendar days.
    """
    return pd.to_datetime(series, utc=True).dt.tz_convert(LOCAL_TZ).dt.date


def _get_date_range(df, params=None):
    """
    Build the daily index used by the dataset.

    The index covers only days present in BOTH sources, i.e. the
    intersection of the tweet dates (local time) and the dispatch-centre
    dates, restricted to days strictly before the truncation date.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tweet DataFrame with a 'created_at' column.
    params : argparse.Namespace, optional
        Uses 'truncation_date' (str, YYYY-MM-DD, exclusive cutoff) and
        'dispatch_dates' (dates present in the dispatch-centre data).
        A missing attribute simply skips the corresponding restriction.
    """
    days = set(tweet_local_dates(df["created_at"]))

    dispatch_dates = getattr(params, "dispatch_dates", None) if params is not None else None
    if dispatch_dates is not None:
        n_before = len(days)
        days &= set(dispatch_dates)
        print(
            f"[_get_date_range] Intersection with dispatch dates: "
            f"{n_before} → {len(days)} days."
        )

    truncation_date = getattr(params, "truncation_date", None) if params is not None else None
    if truncation_date is not None:
        cutoff = pd.to_datetime(truncation_date).date()
        days = {d for d in days if d < cutoff}

    if not days:
        raise ValueError(
            "[_get_date_range] Empty date range: no day is shared by "
            "the tweets and the dispatch data."
        )

    date_range = sorted(days)
    print(
        f"[_get_date_range] Daily index: {date_range[0]} → {date_range[-1]} "
        f"({len(date_range)} days)."
    )
    return date_range


def prepare_bert_style_dataset(df, params):
    """
    Prepare dataset by embedding daily tweet using BERT style model.
    """

    # find days range
    date_range = _get_date_range(df, params)

    # aggregate daily tweet and compute engagement score
    df = aggregate_daily_tweets(df, params)

    # encode all tweet using
    df = per_tweet_encoding(df, params)

    # daily pooling
    df = daily_tweets_pooling(df, params, date_range=date_range)

    df = add_time_series_features(df, params=params)

    return df

def prepare_string_matching_dataset(df, params):
    """"
    Prepare dataset using a fuzzy string matching approach. Specifically, we:
    - extract keywords using a lexicon-based approach
    - construct a daily score by aggregating the keyword counts and
      engagement scores of the tweets from that day.
    
    """
    
    # find days range
    date_range = _get_date_range(df, params)

    # string matching
    df = construct_textual_features(df, params=params)
    
    # daily tweet feature construction
    df = construct_final_tweet_data(df, params=params, date_range=date_range)
    
    df = add_time_series_features(df, params=params)

    return df
    

def add_time_series_features(df: pd.DataFrame, params) -> pd.DataFrame:
    """
    Add lagged versions of daily NLP feature columns to provide the model
    with explicit temporal context across multiple time scales.

    Features added for each NLP column c:
        c_lag1d  : value from day t-1  (immediate momentum)
        c_lag3d  : value from day t-3  (short-term trend)
        c_lag7d  : value from day t-7  (same weekday last week)

    Lagged columns: score_* and count_* (lexicon approaches), n_tweets,
    emb_* (BERT embeddings), and llm_* (LLM relevance counts, added by a
    later stage).

    All lags are strictly causal — lag k at day t uses only information
    available at day t, i.e. the observed value from day t-k.

    Parameters
    ----------
    df : pd.DataFrame
        Daily feature DataFrame with a 'date' column.
        Must be sorted chronologically before calling this function.
    params : argparse.Namespace
        Uses 'nlp_feature_lags' if present (defaults to FEATURE_LAGS).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lag columns appended, sorted by date.
    """
    lags = list(getattr(params, 'nlp_feature_lags', NLP_FEATURE_LAGS))

    # Identify NLP feature columns — score_*/count_* for the lexicon
    # approaches, emb_* for BERT embeddings, llm_* for the LLM relevance
    # counts, plus the daily tweet volume.
    nlp_prefixes = ('score_', 'count_', 'emb_', 'llm_')
    nlp_cols = [
        c for c in df.columns
        if c.startswith(nlp_prefixes) or c == 'n_tweets'
    ]

    if not nlp_cols:
        print("[add_time_series_features] No NLP feature columns found — skipping.")
        return df

    # Ensure chronological order before shifting
    df = df.sort_values('date').reset_index(drop=True)

    n_added = 0
    for lag in lags:
        for col in nlp_cols:
            lag_col = f'{col}_lag{lag}d'
            df[lag_col] = df[col].shift(lag)
            n_added += 1

    print(
        f"[add_time_series_features] Added {n_added} lag features "
        f"({len(nlp_cols)} columns × {len(lags)} lags: {lags})."
    )

    return df

def construct_final_tweet_data(df, params, date_range=None):
    """
    Construct the final daily dataset from the per-tweet lexicon matches.

    Aggregation mode is controlled by params.aggregation:
      - 'count' (default): daily counts.
          n_tweets      : number of tweets that day (all languages)
          count_<entity>: number of tweets that day with >= 1 match for that
                          entity
          count_any     : number of tweets that day with >= 1 match for any
                          entity
          No weighting is applied: engagement was measured after posting and
          is missing for part of the corpus. Days without tweets get 0.
      - 'rate': weighted mean per tweet of the per-tweet match counts, the
        historical behaviour, kept for the sensitivity analysis. The weighting
        is controlled by params.pooling_strategy:
          - 'engagement' : engagement-weighted mean (1 + log1p(engagement))
          - 'mean'       : simple unweighted mean across all tweets per day
    """
    df = df.copy()
    df["date"] = tweet_local_dates(df["created_at"])

    aggregation = getattr(params, 'aggregation', 'count')
    pooling_strategy = getattr(params, 'pooling_strategy', 'mean')

    if aggregation not in ('count', 'rate'):
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Choose from: 'count', 'rate'."
        )

    print(
        f"[construct_final_tweet_data] aggregation={aggregation}"
        + (f", pooling={pooling_strategy}" if aggregation == 'rate' else "")
    )

    if aggregation == 'rate' and pooling_strategy not in ('engagement', 'mean'):
        raise ValueError(
            f"Unknown pooling_strategy '{pooling_strategy}'. "
            f"Choose from: 'engagement', 'mean'."
        )

    # ------------------------------------------------------------------
    # Entity columns
    # ------------------------------------------------------------------
    entity_count_cols = [
        f"text_features_{e}_count"
        for e in ENTITY_LEXICON
        if f"text_features_{e}_count" in df.columns
    ]
    entity_names = [
        c.replace('text_features_', '').replace('_count', '')
        for c in entity_count_cols
    ]

    # ------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------
    if aggregation == 'count':
        daily = pd.DataFrame({"date": df["date"].values})
        for name, col in zip(entity_names, entity_count_cols):
            daily[f"count_{name}"] = (df[col].to_numpy() > 0).astype(int)
        daily["count_any"] = (
            df[entity_count_cols].to_numpy() > 0
        ).any(axis=1).astype(int)
        daily["n_tweets"] = 1

        final = daily.groupby("date").sum()

    else:
        # ----------------------------------------------------------------
        # Weights
        # ----------------------------------------------------------------
        if pooling_strategy == 'engagement':
            engagement_cols = [
                "public_metrics_retweet_count",
                "public_metrics_reply_count",
                "public_metrics_like_count",
                "public_metrics_quote_count",
            ]
            available_engagement = [c for c in engagement_cols if c in df.columns]
            if not available_engagement:
                raise ValueError(
                    "pooling_strategy='engagement' requires engagement columns "
                    "but none were found."
                )
            df["engagement"] = sum(df[c].fillna(0) for c in available_engagement)
            df["weight"] = 1 + np.log1p(df["engagement"])
        else:
            # Mean pooling: uniform weight of 1 for every tweet
            df["weight"] = 1.0

        # --------------------------------------------------------------
        # Entity scores aggregated daily
        # --------------------------------------------------------------
        weighted = df[entity_count_cols].multiply(df["weight"], axis=0).copy()
        weighted["weight"] = df["weight"].values
        weighted["date"] = df["date"].values

        agg = weighted.groupby("date").sum()
        denom = agg["weight"].replace(0, np.nan)
        scores = agg[entity_count_cols].div(denom, axis=0)

        scores.columns = [f"score_{name}" for name in entity_names]

        final = scores.copy()
        final["n_tweets"] = df.groupby("date").size().reindex(final.index).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Assemble final dataframe
    # ------------------------------------------------------------------
    if not final.empty:
        full_range = date_range if date_range is not None else pd.date_range(
            start=min(final.index),
            end=max(final.index),
            freq="D",
        ).date
        final = final.reindex(full_range).fillna(0)

    if aggregation == 'count':
        final[final.columns] = final[final.columns].astype(int)
    elif 'n_tweets' in final.columns:
        final['n_tweets'] = final['n_tweets'].astype(int)

    final = final.sort_index()
    final.index.name = "date"
    final = final.reset_index()

    # ------------------------------------------------------------------
    # Diagnostic summary
    # ------------------------------------------------------------------
    print(
        f"Dataset constructed with {len(final)} days "
        f"and {final.shape[1] - 1} features."
    )

    return final


def truncate_dataset(df: pd.DataFrame, params) -> pd.DataFrame:
    """
    Truncate the dataset to retain only observations strictly before
    the truncation date specified in param.truncation_date (exclusive
    cutoff: the truncation date itself is dropped).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'date' column.
    param : argparse.Namespace
        Configuration object containing a 'truncation_date' attribute
        (str, format: YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Truncated DataFrame containing only rows where 'date' is
        strictly less than param.truncation_date (exclusive cutoff).
    """
    truncation_date = pd.to_datetime(params.truncation_date).date()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mask = df["date"] < truncation_date
    n_before = len(df)
    df = df[mask].reset_index(drop=True)
    n_after = len(df)
    print(f"[truncate_dataset] Truncated dataset from {n_before} to {n_after} rows "
          f"(cutoff: {truncation_date}).")
    return df

def _infer_freq(dates):
    """Return 'D', 'W', or 'M' based on median spacing between dates."""
    if len(dates) < 2:
        return 'D'
    diffs = dates.sort_values().diff().dropna().dt.days
    median = diffs.median()
    if median <= 1:
        return 'D'
    if median <= 10:        # weekly trends are spaced 7 days apart
        return 'W'
    return 'M'               # monthly trends are spaced ~30 days apart


def _load_one(path, region, target_dates, drop_all_zero=True):
    """Load a Google Trends CSV → tidy DataFrame aligned to target_dates."""
    date_col = 'date'
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        print(f"[google_trends] Skipping {path}: {e}")
        return None

    if raw.empty or raw.shape[1] < 2:
        return None

    raw = raw.rename(columns={raw.columns[0]: date_col})
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    raw = raw.dropna(subset=[date_col]).sort_values(date_col)

    feat_cols = [c for c in raw.columns if c != date_col]
    for c in feat_cols:
        raw[c] = (raw[c].astype(str)
                        .str.replace('<', '', regex=False)
                        .replace({'': None}))
        raw[c] = pd.to_numeric(raw[c], errors='coerce')

    rename_map = {
        c: f"google_trends_{region}_{c}".replace(' ', '_').lower()
        for c in feat_cols
    }
    raw = raw.rename(columns=rename_map)

    # --- Frequency-aware alignment ---
    freq = _infer_freq(raw[date_col])
    if freq == 'D':
        return raw  # already daily, plain merge will work

    # Build a daily (or target-frequency) index spanning the target dates,
    # forward-fill the period's value across every day in it.
    full_idx = pd.DataFrame({date_col: pd.to_datetime(target_dates).sort_values().unique()})
    aligned = pd.merge_asof(
        full_idx.sort_values(date_col),
        raw.sort_values(date_col),
        on=date_col,
        direction='backward',  # use the most recent period's value
    )

    # Re-check after alignment: a column may have had non-zero values
    # outside target_dates' range but be entirely zero within it.
    if drop_all_zero:
        feat_cols = [c for c in aligned.columns if c != date_col]
        keep = [c for c in feat_cols if aligned[c].fillna(0).abs().sum() > 0]
        if not keep:
            return None
        aligned = aligned[[date_col] + keep]

    return aligned


def save_dataset(df, args, output_dir):
    """Save the constructed dataset and the params used to create it."""
    output_dir.mkdir(parents=True, exist_ok=True)


    df_path     = output_dir / f"dataset_{args.final_dataset_name}.pkl"
    params_path = output_dir / f"dataset_{args.final_dataset_name}_params.pkl"

    df.to_pickle(df_path)
    pd.to_pickle(vars(args), params_path)

    print(f"[save] Dataset saved to {df_path}  ({len(df)} rows, {df.shape[1]} cols)")
    print(f"[save] Params saved to   {params_path}")


def aggregate_daily_tweets(df, params):
    """
    Prepare a BERT-style dataset with one row per day, containing:
    - date
    - list of all tweet texts from that day
    - list of per-tweet engagement scores (parallel to text)
    - list of per-tweet languages (parallel to text)

    If params.considered_language is not 'all', only tweets in the
    specified language are retained before aggregation.

    Language mapping:
        'all'     → no filtering
        'french'  → lang == 'fr'
        'german'  → lang == 'de'
        'english' → lang == 'en'
    """
    LANGUAGE_MAP = {
        'french':  'fr',
        'german':  'de',
        'english': 'en',
    }

    engagement_cols = [
        "public_metrics_retweet_count",
        "public_metrics_reply_count",
        "public_metrics_like_count",
        "public_metrics_quote_count",
    ]
    available_engagement = [c for c in engagement_cols if c in df.columns]
    if not available_engagement:
        raise ValueError("No engagement columns found.")

    # Keep lang column for filtering and diagnostics
    df = df[['created_at', 'text', 'lang', 'hashtags'] + available_engagement].copy()
    df["date"] = tweet_local_dates(df["created_at"])
    df["engagement"] = sum(df[c].fillna(0) for c in available_engagement)

    # Language filtering
    considered_language = params.considered_language
    if considered_language != 'all':
        if considered_language not in LANGUAGE_MAP:
            raise ValueError(f"Unknown language '{considered_language}'. "f"Valid options: {['all'] + list(LANGUAGE_MAP.keys())}")
        lang_code = LANGUAGE_MAP[considered_language]
        n_before = len(df)
        df = df[df['lang'] == lang_code].copy()
        n_after = len(df)
        print(
            f"[aggregate_daily_tweets] Language filter '{considered_language}' "
            f"({lang_code}): {n_before} → {n_after} tweets "
            f"({n_after/n_before:.1%} retained)"
        )
    else:
        print(
            f"[aggregate_daily_tweets] No language filter applied "
            f"({len(df)} tweets retained)"
        )

    # Diagnostic: language distribution after filtering
    lang_counts = df['lang'].value_counts()
    print(f"[aggregate_daily_tweets] Language distribution after filtering:")
    for lang, count in lang_counts.head(5).items():
        print(f"  {lang}: {count} ({count/len(df):.1%})")

    # Group by date into parallel lists
    df_grouped = df.groupby('date').agg({
        'text':       list,
        'engagement': list,
        'lang':       list,
    }).reset_index()

    print(
        f"[aggregate_daily_tweets] {len(df_grouped)} days, "
        f"mean {df_grouped['text'].apply(len).mean():.1f} tweets/day, "
        f"min {df_grouped['text'].apply(len).min()}, "
        f"max {df_grouped['text'].apply(len).max()}"
    )

    return df_grouped


def per_tweet_encoding(df, params, batch_size=32, max_length=512, pooling='cls'):
    """
    Encode each tweet independently using a BERT-style model.

    Adds an 'embeddings' column to df: each row contains a numpy array of
    shape (n_tweets_that_day, hidden_size) with one embedding per tweet.

    Expected keys in params:
        - model_name (str): HuggingFace model identifier
                            (e.g. 'ZurichNLP/swissbert')
        - batch_size (int, default 32)
        - max_length (int, default 512)
        - pooling (str, default 'cls'): 'cls' or 'mean'
        - device (str, optional): 'cuda', 'cpu', or 'mps';
                                  auto-detected if absent
    """
    model_name = params.bert_embedding_model
    device =  'cuda' if torch.cuda.is_available() else 'cpu'

    print('The model we are using is ', device)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Flatten all tweets across all days into one list, remembering
    # how many tweets belong to each day so we can split back afterwards.
    all_tweets = []
    day_lengths = []
    for tweets in df['text']:
        all_tweets.extend(tweets)
        day_lengths.append(len(tweets))

    # Encode in batches
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_tweets), batch_size), desc='Encoding tweets'):
            batch = all_tweets[i:i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
            ).to(device)

            out = model(**enc)
            hidden = out.last_hidden_state  # (B, T, H)

            if pooling == 'cls':
                vecs = hidden[:, 0, :]
            elif pooling == 'mean':
                mask = enc['attention_mask'].unsqueeze(-1).float()
                vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_embeddings.append(vecs.detach().cpu().clone().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # Split back into per-day arrays
    df = df.copy()
    splits = np.cumsum(day_lengths)[:-1]
    df['embeddings'] = np.split(all_embeddings, splits)

    return df



def daily_tweets_pooling(df, params, date_range=None):
    """
    Pool per-tweet embeddings into a single daily vector, then expand
    into one column per embedding dimension.

    Reads the 'embeddings' column (each row: array of shape (n_tweets, H))
    and writes columns 'emb_0', 'emb_1', ..., 'emb_{H-1}', plus 'n_tweets'
    (the number of tweets pooled that day).

    Expected keys in params:
        - pooling_strategy (str): 'mean', 'max', or 'engagement'
        - min_class_volume (int): days with fewer tweets get NaN
    """
    strategy = params.pooling_strategy

    df = df.copy()
    daily_vectors = []

    for idx, row in df.iterrows():
        emb = row['embeddings']  # (n_tweets, H)

        if strategy == 'mean':
            daily = emb.mean(axis=0)
        elif strategy == 'max':
            daily = emb.max(axis=0)
        elif strategy == 'engagement':
            engagement = np.asarray(row['engagement'], dtype=float)
            weights = 1.0 + np.log1p(engagement)
            weights = weights / weights.sum()
            daily = (emb * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

        daily_vectors.append(daily)

     # Expand into one column per embedding dimension
    daily_matrix = np.vstack(daily_vectors)
    emb_df = pd.DataFrame(
        daily_matrix,
        index=df.index,
        columns=[f"emb_{i}" for i in range(daily_matrix.shape[1])],
    )

    # Number of tweets pooled into each daily vector
    n_tweets = pd.DataFrame(
        {"n_tweets": df["text"].apply(len).astype(int).to_numpy()},
        index=df.index,
    )

    # Build output: date + daily tweet volume + expanded embedding columns
    result = pd.concat(
        [df[['date']].reset_index(drop=True),
         n_tweets.reset_index(drop=True),
         emb_df.reset_index(drop=True)],
        axis=1
    )

    # Reindex to canonical date range: NaN embeddings for missing dates,
    # but the tweet volume of a day without tweets is 0, not NaN.
    if date_range is not None:
        result = result.set_index('date')
        result = result.reindex(date_range)
        result.index.name = 'date'
        result["n_tweets"] = result["n_tweets"].fillna(0).astype(int)
        result = result.reset_index()

    print(f"[daily_tweets_pooling] Output shape: {result.shape}")
    print(f"[daily_tweets_pooling] NaN rows: {result.isnull().any(axis=1).sum()}")

    return result