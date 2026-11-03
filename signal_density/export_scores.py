"""Export LLM scores to a flat CSV for annotation and downstream analysis.

One row per scored tweet: ``tweet_id, day, content, p_yes, informative, lang,
in_geneva, status``. The labels parquet stores no text, so the content is
re-joined from the tweet corpus and passed through the same preprocessing the
model saw — an annotator then judges exactly the string that was scored
(same convention as ``export_pilot.py``). Pass ``--raw`` for the untouched tweet.

Rows whose status is not ``ok`` get an empty probability, so a failed call can
never be mistaken for a negative label.

Usage:
    ../.venv/bin/python export_scores.py --input outputs/llm_labels.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_density import text_preprocessing, tweet_loader  # noqa: E402

OUT_COLUMNS = [
    "tweet_id",
    "day",
    "content",
    "p_yes",
    "informative",
    "lang",
    "in_geneva",
    "status",
]


def export(input_path: Path, output_path: Path, *, raw: bool = False) -> dict:
    labels = pd.read_parquet(input_path).drop_duplicates("tweet_id", keep="first")
    labels["tweet_id"] = labels["tweet_id"].astype(str)

    corpus = tweet_loader.load_tweets(max_distance_km=None, verbose=False)
    corpus = corpus.drop_duplicates("tweet_id", keep="first")
    corpus["tweet_id"] = corpus["tweet_id"].astype(str)
    texts = dict(zip(corpus["tweet_id"], corpus["text"]))

    out = pd.DataFrame(index=labels.index)
    out["tweet_id"] = labels["tweet_id"]
    out["day"] = pd.to_datetime(labels["date_local"]).dt.strftime("%Y-%m-%d")
    raw_text = labels["tweet_id"].map(texts)
    if raw_text.isna().all():
        raise SystemExit(f"No tweet_id from {input_path} found in the corpus.")
    if raw_text.isna().any():
        missing = int(raw_text.isna().sum())
        print(f"Warning: {missing} labelled tweet_ids are absent from the corpus.")
    out["content"] = (
        raw_text
        if raw
        else raw_text.fillna("").map(text_preprocessing.preprocess_tweet_text)
    ).fillna("")
    ok = labels["status"].astype(str).eq("ok")
    out["p_yes"] = labels["p_informative"].astype(float).where(ok)
    out["informative"] = labels["informative_hard"].astype(bool) & ok
    out["lang"] = labels["lang"].astype(str)
    out["in_geneva"] = labels["in_geneva"]
    out["status"] = labels["status"].astype(str)

    out = out[OUT_COLUMNS].sort_values(["day", "tweet_id"], kind="mergesort")
    out = out.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return {
        "rows": len(out),
        "status_ok": int(ok.sum()),
        "informative": int(out["informative"].sum()),
        "empty_content": int(out["content"].eq("").sum()),
        "days": int(out["day"].nunique()),
        "date_range": f"{out['day'].min()} .. {out['day'].max()}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("signal_density/outputs/llm_labels.parquet"),
        help="Parquet written by classify_tweets.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input_dir>/tweet_scores.csv",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Export the untouched tweet text instead of the preprocessed text "
        "the model was given.",
    )
    args = parser.parse_args()
    out_path = args.output or args.input.parent / "tweet_scores.csv"
    stats = export(args.input, out_path, raw=args.raw)
    print(f"Wrote {out_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
