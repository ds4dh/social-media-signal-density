"""Export a random sample of tweets for double human annotation.

Two sheets are written to ``human_review/``:

* ``annotation_sheet.csv`` — ``tweet_id, text, FH, LT``. This is the blind sheet
  handed to the two annotators: no model answer anywhere, and the two empty
  columns are filled in by hand (FH = the author, LT = Léonard).
* ``annotation_sheet_with_llm.csv`` — the same rows in the same order, with the
  LLM annotation appended (``llm_p_informative``, ``llm_informative``,
  ``llm_p_yes``, ``llm_p_no``, ``llm_status``, plus ``day``, ``lang`` and
  ``in_geneva`` for diagnostics). This copy exists only to compute human/LLM
  agreement after both columns are filled; it is not to be shown to the
  annotators.

Input: the labels written by ``classify_tweets.py`` (``llm_labels.parquet``).
That file can hold several runs (prompt x model); exactly one run is selected,
by ``--prompt_hash`` (prefix) when the file holds more than one. The parquet
file carries no text, so the tweet text is joined from the corpus loaded with
``tweet_loader.load_tweets``: the same ``text`` column ``classify_tweets.py``
passes to the prompt, so annotators judge the same input as the LLM.

Sampling is uniform over the unique ``tweet_id``s of the selected run, drawn
after sorting by ``tweet_id`` so ``--seed`` is reproducible regardless of row
order. Dropped by default:

* tweets whose text is empty (URL/mention-only tweets);
* the 50 pilot tweets (already read during prompt development, scored with a
  different prompt); ``--include-pilot`` keeps them.

Existing sheets are never overwritten unless ``--overwrite`` is passed, so
filled-in annotations cannot be lost by re-running the script.

Usage:
    ../.venv/bin/python export_human_review.py [--prompt_hash a6224977]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_density import tweet_loader  # noqa: E402

OUTPUT_DIR = PACKAGE_DIR / "human_review"
DEFAULT_INPUT = PACKAGE_DIR / "outputs" / "llm_labels.parquet"
PILOT_LABELS_FILE = PACKAGE_DIR / "outputs" / "pilot_llm_labels.csv"

SHEET_FILE = "annotation_sheet.csv"
SHEET_WITH_LLM_FILE = "annotation_sheet_with_llm.csv"
PARAMS_FILE = "annotation_sheet_params.json"

SAMPLE_SIZE = 200
SEED = 42
ANNOTATOR_COLUMNS = ["FH", "LT"]

SHEET_COLUMNS = ["tweet_id", "text", *ANNOTATOR_COLUMNS]
LLM_COLUMNS = [
    "llm_p_informative", "llm_informative", "llm_p_yes", "llm_p_no",
    "llm_status", "day", "lang", "in_geneva",
]


def load_labels(input_path: Path, prompt_hash: str | None) -> pd.DataFrame:
    """Labels of exactly one run (prompt x model) of ``classify_tweets.py``."""
    df = pd.read_parquet(input_path)
    df["tweet_id"] = df["tweet_id"].astype(str)
    if prompt_hash:
        df = df[df["prompt_hash"].str.startswith(prompt_hash)]
        if df.empty:
            raise SystemExit(f"No labels with prompt_hash starting with {prompt_hash!r} in {input_path}.")
    runs = df[["prompt_hash", "model_fingerprint"]].drop_duplicates()
    if len(runs) != 1:
        listing = "\n".join(
            f"  prompt_hash={r.prompt_hash[:16]}…  n={int(((df.prompt_hash == r.prompt_hash) & (df.model_fingerprint == r.model_fingerprint)).sum())}"
            for r in runs.itertuples()
        )
        raise SystemExit(f"{len(runs)} runs in {input_path}; select one with --prompt_hash:\n{listing}")
    return df.drop_duplicates("tweet_id", keep="last")


def load_texts(tweets_pkl: Path | None) -> pd.DataFrame:
    """``tweet_id -> text`` as passed to the prompt by ``classify_tweets.py``."""
    tweets = tweet_loader.load_tweets(
        tweets_pkl, cutoff=tweet_loader.CUTOFF_LOCAL_DATE, max_distance_km=None
    )
    tweets = tweets[["tweet_id", "text"]].copy()
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)
    return tweets.drop_duplicates("tweet_id", keep="first")


def load_pool(input_path: Path, prompt_hash: str | None, tweets_pkl: Path | None) -> pd.DataFrame:
    """One row per labelled tweet with its text, ready to be sampled from."""
    labels = load_labels(input_path, prompt_hash)
    df = labels.merge(load_texts(tweets_pkl), on="tweet_id", how="left", validate="one_to_one")
    missing = df["text"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} labelled tweets have no text in the corpus; check --tweets_pkl.")
    blank = df["text"].astype(str).str.strip().eq("")
    df = df[~blank]
    return df.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)


def pilot_tweet_ids(pilot_path: Path) -> set[str]:
    """Tweet ids of the annotation pilot, empty if that file is absent."""
    if not pilot_path.exists():
        return set()
    return set(pd.read_csv(pilot_path, dtype={"tweet_id": str})["tweet_id"])


def sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` rows without replacement from a deterministically sorted pool."""
    if n > len(pool):
        raise SystemExit(f"Cannot draw {n} tweets from a pool of {len(pool)}.")
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


def build_frames(chosen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame({"tweet_id": chosen["tweet_id"].to_numpy(), "text": chosen["text"].to_numpy()})
    for col in ANNOTATOR_COLUMNS:
        base[col] = ""
    sheet = base.copy()
    with_llm = base.copy()
    with_llm["llm_p_informative"] = chosen["p_informative"].to_numpy()
    with_llm["llm_informative"] = chosen["informative_hard"].astype("boolean").astype("Int64").to_numpy()
    with_llm["llm_p_yes"] = chosen["p_yes"].to_numpy()
    with_llm["llm_p_no"] = chosen["p_no"].to_numpy()
    with_llm["llm_status"] = chosen["status"].to_numpy()
    with_llm["day"] = chosen["date_local"].to_numpy()
    with_llm["lang"] = chosen["lang"].to_numpy()
    with_llm["in_geneva"] = chosen["in_geneva"].to_numpy()
    return sheet[SHEET_COLUMNS], with_llm[["tweet_id", "text", *ANNOTATOR_COLUMNS, *LLM_COLUMNS]]


def export(
    input_path: Path,
    output_dir: Path,
    *,
    n: int = SAMPLE_SIZE,
    seed: int = SEED,
    prompt_hash: str | None = None,
    tweets_pkl: Path | None = None,
    include_pilot: bool = False,
    pilot_path: Path = PILOT_LABELS_FILE,
    overwrite: bool = False,
) -> dict:
    sheet_path = output_dir / SHEET_FILE
    with_llm_path = output_dir / SHEET_WITH_LLM_FILE
    existing = [p for p in (sheet_path, with_llm_path) if p.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite (filled-in annotations would be lost): "
            + ", ".join(map(str, existing)) + ". Use --overwrite or another --output_dir."
        )

    pool = load_pool(input_path, prompt_hash, tweets_pkl)
    excluded = set() if include_pilot else pilot_tweet_ids(pilot_path)
    excluded &= set(pool["tweet_id"])
    if excluded:
        pool = pool[~pool["tweet_id"].isin(excluded)].reset_index(drop=True)
    chosen = sample(pool, n, seed)
    sheet, with_llm = build_frames(chosen)

    output_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(sheet_path, index=False)
    with_llm.to_csv(with_llm_path, index=False)

    stats = {
        "input": str(input_path),
        "prompt_hash": str(chosen["prompt_hash"].iloc[0]),
        "model_fingerprint": json.loads(chosen["model_fingerprint"].iloc[0]),
        "n_pool": len(pool),
        "n_excluded_pilot": len(excluded),
        "n_sampled": len(sheet),
        "seed": seed,
        "status_counts_sampled": with_llm["llm_status"].value_counts().to_dict(),
        "positives_sampled": int(with_llm["llm_informative"].fillna(0).sum()),
        "mean_p_informative_sampled": float(pd.to_numeric(with_llm["llm_p_informative"]).mean()),
        "date_range": f"{with_llm['day'].min()} .. {with_llm['day'].max()}",
        "sheet": str(sheet_path),
        "sheet_with_llm": str(with_llm_path),
    }
    (output_dir / PARAMS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt_hash", default=None,
                        help="Prefix of the prompt hash of the run to sample (needed if the file holds several runs).")
    parser.add_argument("--tweets_pkl", type=Path, default=None, help="Default: DATA_DIR/tweets.pkl")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--include-pilot", action="store_true", help="Keep the 50 pilot tweets in the sampling pool.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation sheets.")
    args = parser.parse_args()
    stats = export(
        args.input, args.output_dir, n=args.n, seed=args.seed, prompt_hash=args.prompt_hash,
        tweets_pkl=args.tweets_pkl, include_pilot=args.include_pilot, overwrite=args.overwrite,
    )
    print(f"Wrote {stats['sheet']}")
    print(f"Wrote {stats['sheet_with_llm']}")
    for key in ("prompt_hash", "n_pool", "n_excluded_pilot", "n_sampled", "seed",
                "status_counts_sampled", "positives_sampled", "mean_p_informative_sampled", "date_range"):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()