"""Per-tweet feature cache.

Each tweet's features are computed once, on unique tweets:

- ``lexicon``  per-tweet entity indicators (1 if the tweet matches the entity),
  using the unchanged matcher of ``Dataset construction/textual_features.py``.
- ``bert``     per-tweet TwHIN-BERT embedding, same model, CLS pooling,
  ``max_length`` and tokenisation call as ``final_dataset_construction.per_tweet_encoding``.
- ``llm``      ``informative``, ``p_yes`` and ``status`` of the full-corpus run.

Both computed caches are resumable: tweets that are already cached are skipped,
and a BERT chunk is only published once its ``.npy`` and its id file are both on
disk.

Run: ``python build_tweet_cache.py [--skip-bert] [--chunksize N]``
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from corpus import check_prompt_hash, load_llm_labels, load_unique_tweets
from paths import (
    BERT_BATCH_SIZE,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_IDS,
    BERT_MATRIX,
    CACHE_DIR,
    LEXICON_CACHE,
    LLM_CACHE,
)

#: directory holding finished BERT chunks before consolidation
BERT_PARTS_DIR = CACHE_DIR / "_bert_parts"


# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------

def per_tweet_entity_indicators(tweets: pd.DataFrame) -> pd.DataFrame:
    """Return ``tweet_id`` plus one 0/1 column per entity for ``tweets``.

    ``construct_textual_features`` is called unchanged, so the matching rules
    (normalisation, per-language lexicons, adaptive fuzzy cutoff, overlap
    resolution) are exactly the real pipeline's. The pipeline counts matches per
    tweet, while its daily count counts tweets with at least one match, so the
    indicator ``count > 0`` is the lossless per-tweet form of it.
    """
    from constants import ENTITY_LEXICON
    from textual_features import construct_textual_features

    df = tweets[["tweet_id", "text", "lang"]].copy()
    df = construct_textual_features(df, params=None)
    out = df[["tweet_id"]].copy()
    for entity in ENTITY_LEXICON:
        col = f"text_features_{entity}_count"
        if col in df.columns:
            out[entity] = (df[col].to_numpy() > 0).astype("int8")
    return out


def build_lexicon_cache(
    tweets: pd.DataFrame,
    out_path: Path = LEXICON_CACHE,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute (or reuse) the per-tweet entity indicators, keyed by ``tweet_id``.

    Resumable: ids already in ``out_path`` are skipped, the new rows are
    appended, and the file is rewritten sorted by ``tweet_id``.
    """
    tweets = tweets.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)
    cached = out_path if out_path.exists() else None
    cached_df = pd.read_parquet(cached) if cached is not None else None
    todo = (
        tweets
        if cached_df is None
        else tweets[~tweets["tweet_id"].isin(cached_df["tweet_id"])]
    )
    if verbose:
        n_cached = 0 if cached_df is None else len(cached_df)
        print(f"[build_lexicon_cache] {len(tweets)} tweets, {n_cached} cached, "
              f"{len(todo)} to compute.")
    if len(todo):
        fresh = per_tweet_entity_indicators(todo)
        cached_df = (
            fresh if cached_df is None
            else pd.concat([cached_df, fresh], ignore_index=True)
        )
        cached_df = (
            cached_df.drop_duplicates("tweet_id", keep="last")
            .sort_values("tweet_id", kind="mergesort")
            .reset_index(drop=True)
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cached_df.to_parquet(out_path, index=False)
    if verbose:
        per_entity = cached_df.drop(columns=["tweet_id"]).sum(axis=0).astype(int).to_dict()
        print(f"[build_lexicon_cache] cache: {len(cached_df)} tweets, "
              f"{cached_df.shape[1] - 1} entities. Matching tweets per entity: "
              f"{per_entity}")
    return cached_df


# ---------------------------------------------------------------------------
# BERT
# ---------------------------------------------------------------------------

def bert_chunk_files() -> list[tuple[int, Path, Path]]:
    """Return ``(chunk_index, npy_path, ids_path)`` for every complete chunk."""
    if not BERT_PARTS_DIR.exists():
        return []
    parts = []
    for npy in sorted(BERT_PARTS_DIR.glob("part_*.npy")):
        ids = npy.with_name(npy.stem + "_ids.parquet")
        if ids.exists():
            parts.append((int(npy.stem.split("_")[1]), npy, ids))
    return parts


def write_bert_chunk(vectors: np.ndarray, ids: list[str], chunk_index: int) -> Path:
    """Publish one BERT chunk atomically: both files present, or neither."""
    BERT_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    npy = BERT_PARTS_DIR / f"part_{chunk_index:05d}.npy"
    ids_path = BERT_PARTS_DIR / f"part_{chunk_index:05d}_ids.parquet"
    tmp_npy = BERT_PARTS_DIR / f"tmp_part_{chunk_index:05d}"
    tmp_ids = ids_path.with_name(ids_path.name + ".tmp")
    np.save(tmp_npy, np.asarray(vectors, dtype="float32"))  # appends ".npy"
    shutil.move(str(tmp_npy) + ".npy", npy)
    pd.DataFrame({"tweet_id": [str(i) for i in ids]}).to_parquet(tmp_ids, index=False)
    shutil.move(tmp_ids, ids_path)
    return npy


def consolidate_bert(matrix_path: Path = BERT_MATRIX, ids_path: Path = BERT_IDS) -> pd.DataFrame:
    """Concatenate the published chunks into ``bert.npy`` + ``bert_ids.parquet``.

    Duplicate ids keep their first occurrence. Rows are sorted by ``tweet_id`` so
    the row order is reproducible. Returns the id table (row order of the matrix).
    """
    parts = bert_chunk_files()
    if not parts:
        raise FileNotFoundError(f"No BERT chunks in {BERT_PARTS_DIR}.")
    seen: set[str] = set()
    matrices: list[np.ndarray] = []
    ids: list[str] = []
    for _, npy, ids_file in parts:
        chunk_ids = pd.read_parquet(ids_file)["tweet_id"].tolist()
        arr = np.load(npy)
        keep = [i for i, t in enumerate(chunk_ids) if t not in seen]
        seen.update(chunk_ids[i] for i in keep)
        matrices.append(arr[keep])
        ids.extend(chunk_ids[i] for i in keep)
    matrix = np.concatenate(matrices, axis=0)
    order = np.argsort(np.asarray(ids, dtype=object), kind="mergesort")
    matrix = matrix[order]
    ids = [ids[i] for i in order]
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    table = pd.DataFrame({"tweet_id": ids})
    table.to_parquet(ids_path, index=False)
    print(f"[consolidate_bert] {len(table)} embeddings of dim {matrix.shape[1]} "
          f"-> {matrix_path.name} ({matrix.nbytes / 1e6:.0f} MB).")
    return table


def load_bert_cache(matrix_path: Path = BERT_MATRIX, ids_path: Path = BERT_IDS):
    """Return ``(matrix, ids)``; consolidates the chunks if the matrix is absent."""
    if not matrix_path.exists() or not ids_path.exists():
        consolidate_bert(matrix_path, ids_path)
    return np.load(matrix_path), pd.read_parquet(ids_path)["tweet_id"].tolist()


def build_bert_cache(
    tweets: pd.DataFrame,
    *,
    model_name: str = BERT_MODEL_NAME,
    batch_size: int = BERT_BATCH_SIZE,
    max_length: int = BERT_MAX_LENGTH,
    chunksize: int = 2000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Embed every uncached tweet as the CLS vector of the TwHIN-BERT model.

    Resumable: ids already published (consolidated matrix or chunk) are skipped,
    and chunk numbering continues from the published count, so re-running
    appends instead of overwriting.
    """
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    tweets = tweets.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)
    cached_ids: set[str] = set()
    if BERT_MATRIX.exists() and BERT_IDS.exists():
        cached_ids |= set(pd.read_parquet(BERT_IDS)["tweet_id"])
    for _, _, ids_file in bert_chunk_files():
        cached_ids |= set(pd.read_parquet(ids_file)["tweet_id"])
    todo = tweets[~tweets["tweet_id"].isin(cached_ids)].reset_index(drop=True)
    n_chunks = len(bert_chunk_files())
    if verbose:
        print(f"[build_bert_cache] {len(tweets)} tweets, {len(cached_ids)} cached, "
              f"{len(todo)} to embed ({n_chunks} chunks already published).")
    if not len(todo):
        if not (BERT_MATRIX.exists() and BERT_IDS.exists()):
            return consolidate_bert()
        return pd.read_parquet(BERT_IDS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if verbose:
        print(f"[build_bert_cache] model={model_name} device={device} "
              f"batch_size={batch_size} max_length={max_length} pooling=cls")

    texts = todo["text"].astype(str).tolist()
    ids = todo["tweet_id"].tolist()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), chunksize), desc="Embedding tweets"):
            vectors = []
            for i in range(start, min(start + chunksize, len(texts)), batch_size):
                enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state
                vectors.append(hidden[:, 0, :].detach().cpu().clone().numpy())
            write_bert_chunk(np.concatenate(vectors, axis=0),
                             ids[start:start + chunksize], n_chunks)
            n_chunks += 1
    return consolidate_bert()


# ---------------------------------------------------------------------------
# LLM labels
# ---------------------------------------------------------------------------

def build_llm_cache(out_path: Path = LLM_CACHE) -> pd.DataFrame:
    """Copy the full-corpus LLM labels into the cache and check the prompt hash."""
    prompt_hash = check_prompt_hash()
    llm = load_llm_labels(verbose=False)
    out = llm[["tweet_id", "day", "p_yes", "informative", "status", "lang"]].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[build_llm_cache] {len(out)} rows -> {out_path.name} "
          f"(prompt_hash {prompt_hash[:12]}).")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the per-tweet feature cache.")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Build only the lexicon and LLM caches.")
    parser.add_argument("--chunksize", type=int, default=2000,
                        help="Tweets per published BERT chunk (resume granularity).")
    parser.add_argument("--batch_size", type=int, default=BERT_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tweets = load_unique_tweets()
    build_llm_cache()
    build_lexicon_cache(tweets)
    if args.skip_bert:
        print("[main] --skip-bert: BERT cache not built.")
        return
    build_bert_cache(tweets, batch_size=args.batch_size, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
