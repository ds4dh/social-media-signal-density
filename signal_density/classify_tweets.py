"""Label every tweet with the local LLM and store the answer probabilities.

For each tweet the prompt of ``prompts/<prompt>.txt`` is sent once, with
``temperature=0`` and ``max_tokens=1``, and the 20 top logprobs of the single
generated token are read back. Tokens that spell ``Yes``/``No`` after stripping
whitespace and punctuation are grouped per answer, their probabilities summed,
and ``p_informative = p_yes / (p_yes + p_no)``. When neither answer shows up in
the top 20 the row gets ``status="no_answer"`` and a NaN score; when the request
keeps failing after all retries it gets ``status="error"``.

Run ``--dry_run`` first to write the parameter file and the prompt without
touching the server.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signal_density import llm_client, text_preprocessing, tweet_loader  # noqa: E402
from signal_density.llm_client import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SEED,
    DEFAULT_TIMEOUT,
    STATUS_ERROR,
    STATUS_NO_ANSWER,
    STATUS_OK,
    LLMConfigError,
)

OUTPUT_DIR = PACKAGE_DIR / "outputs"
LABELS_FILE = "llm_labels.parquet"
PARAMS_FILE = "llm_labels_params.json"
BATCH_SIZE = 500

OUTPUT_COLUMNS = [
    "tweet_id",
    "date_local",
    "lang",
    "in_geneva",
    "p_yes",
    "p_no",
    "p_informative",
    "informative_hard",
    "status",
    "prompt_hash",
    "model_fingerprint",
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.string()),
        ("date_local", pa.string()),
        ("lang", pa.string()),
        ("in_geneva", pa.bool_()),
        ("p_yes", pa.float64()),
        ("p_no", pa.float64()),
        ("p_informative", pa.float64()),
        ("informative_hard", pa.bool_()),
        ("status", pa.string()),
        ("prompt_hash", pa.string()),
        ("model_fingerprint", pa.string()),
    ]
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tweets_pkl", type=Path, default=None, help="Default: DATA_DIR/tweets.pkl")
    parser.add_argument("--limit", type=int, default=None, help="Classify only the first N tweets.")
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Classify a random sample of N distinct tweets, drawn with --seed. "
        "Use this instead of --limit: --limit takes the oldest N tweets of the "
        "corpus, which is one early calendar window.",
    )
    parser.add_argument("--prompt", default=text_preprocessing.DEFAULT_PROMPT)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(llm_client.BACKENDS))
    parser.add_argument("--max_concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--cutoff", default=str(tweet_loader.CUTOFF_LOCAL_DATE.date()),
        help="Keep tweets with a local date before this date.",
    )
    parser.add_argument(
        "--max_distance_km", type=float, default=350.0,
        help="Drop located tweets further than this from Geneva (None disables).",
    )
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore existing labels and classify the whole selection again.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Write the parameter file and stop without calling the server.",
    )
    return parser.parse_args(argv)


def model_fingerprint_json(backend: llm_client.BackendConfig, prompt_hash: str, seed: int) -> str:
    """Fingerprint stored next to every label, identifying a full run."""
    payload = dict(backend.fingerprint)
    payload.update(
        {"base_url": backend.base_url, "backend": backend.backend, "seed": int(seed)}
    )
    payload["prompt_hash"] = prompt_hash
    return json.dumps(payload, sort_keys=True)


def describe_selection(args) -> str:
    """How the tweets to classify were chosen, for the params file."""
    if args.sample is not None:
        return f"random_sample(n={args.sample}, seed={args.seed}): {tweet_loader.SAMPLING_DESCRIPTION}"
    if args.limit is not None:
        return f"head({args.limit}) of the tweet_id-sorted corpus"
    return "full corpus"


def params_payload(args, prompt_hash: str, backend: llm_client.BackendConfig, n_total: int, n_pending: int) -> dict:
    return {
        "script": "classify_tweets.py",
        "seed": args.seed,
        "backend": backend.name,
        "model_fingerprint": json.loads(backend.fingerprint_json()),
        "base_url": backend.base_url,
        "prompt_template": args.prompt,
        "prompt_hash": prompt_hash,
        "temperature": 0.0,
        "max_tokens": 1,
        "top_logprobs": backend.top_logprobs,
        "thinking": backend.thinking,
        "max_concurrency": args.max_concurrency,
        "max_retries": args.max_retries,
        "timeout_s": args.timeout,
        "batch_size": args.batch_size,
        "cutoff_local_date": args.cutoff,
        "max_distance_km": args.max_distance_km,
        "local_timezone": tweet_loader.LOCAL_TZ,
        "tweets_pkl": str(args.tweets_pkl) if args.tweets_pkl else str(tweet_loader.DATA_DIR / "tweets.pkl"),
        "n_tweets_selected": n_total,
        "n_tweets_pending": n_pending,
        "selection": describe_selection(args),
        "answer_token_mapping": "strip whitespace+punctuation, lowercase; yes/y -> Yes, no/n -> No",
    }


def stored_labels(path: Path) -> pd.DataFrame:
    """Every previously written label row, empty frame if there is no output yet."""
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    stored = pq.read_table(path, schema=OUTPUT_SCHEMA).to_pandas()
    return stored[OUTPUT_COLUMNS]


def labelled_ids(stored: pd.DataFrame, prompt_hash: str, fingerprint: str) -> set[str]:
    """Tweets already scored by this exact prompt+model run.

    Rows from a different prompt or model are excluded, and so are ``error``
    rows, which a resumed run retries. ``no_answer`` rows are kept: at
    temperature 0 the same prompt yields the same answer, so retrying them
    would only repeat a wasted request.
    """
    if stored.empty:
        return set()
    ours = stored[
        (stored["prompt_hash"] == prompt_hash) & (stored["model_fingerprint"] == fingerprint)
    ]
    return set(ours.loc[ours["status"].isin([STATUS_OK, STATUS_NO_ANSWER]), "tweet_id"])


def arrow_table(frame: pd.DataFrame):
    """Arrow table of the output columns, ready to append to the parquet file.

    The schema is applied explicitly because pandas' ``str`` dtype would
    otherwise map to ``large_string`` and break against an existing file.
    """
    return pa.Table.from_pandas(frame[OUTPUT_COLUMNS], schema=OUTPUT_SCHEMA, preserve_index=False)


def write_labels(path: Path, batches: list) -> None:
    """Replace the labels file atomically with ``batches``.

    ``ParquetWriter`` only emits its footer when it is closed, so a run killed
    halfway leaves a file that cannot be read at all and every score of that run
    is lost (this destroyed a 12,000-tweet full run). Rewriting through a
    temporary file and renaming it keeps the last complete file on disk at all
    times, at the cost of rewriting a sub-megabyte file once per batch.
    """
    table = pa.Table.from_batches(batches, schema=OUTPUT_SCHEMA)
    tmp_path = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, path)


def previous_tables(path: Path, prompt_hash: str, fingerprint: str, drop_ids: set[str]):
    """Batches of an existing output file, minus the rows about to be redone.

    Only this run's failed rows are dropped, so labels made earlier with a
    different prompt or model survive untouched alongside the new ones.
    """
    if not path.exists():
        return []
    stored = stored_labels(path)
    stale = (
        (stored["prompt_hash"] == prompt_hash)
        & (stored["model_fingerprint"] == fingerprint)
        & stored["tweet_id"].isin(drop_ids)
    )
    stored = stored[~stale]
    if stored.empty:
        return []
    return list(arrow_table(stored).to_batches())


def results_frame(batch: pd.DataFrame, results: list[dict], prompt_hash: str, fingerprint: str) -> pd.DataFrame:
    """Combine one tweet batch and its API answers into output rows."""
    out = batch[["tweet_id", "date_local", "lang", "in_geneva"]].copy()
    answers = pd.DataFrame(results)
    # Masked dtypes keep a missing score as a parquet null rather than a NaN.
    out["p_yes"] = pd.array(answers["p_yes"], dtype="Float64")
    out["p_no"] = pd.array(answers["p_no"], dtype="Float64")
    scores = pd.array(answers["p_informative"], dtype="Float64")
    out["p_informative"] = scores
    out["informative_hard"] = (scores >= 0.5).astype("boolean")
    out["status"] = answers["status"].astype(str).to_numpy()
    out["prompt_hash"] = prompt_hash
    out["model_fingerprint"] = fingerprint
    return out[OUTPUT_COLUMNS]


def summarise(frame: pd.DataFrame) -> dict:
    counts = frame["status"].value_counts().to_dict() if len(frame) else {}
    scored = pd.to_numeric(frame["p_informative"], errors="coerce").dropna() if len(frame) else pd.Series(dtype=float)
    hard = pd.to_numeric(frame["informative_hard"], errors="coerce") if len(frame) else pd.Series(dtype=float)
    return {
        "n_rows": int(len(frame)),
        "n_done": int(counts.get(STATUS_OK, 0)),
        "n_no_answer": int(counts.get(STATUS_NO_ANSWER, 0)),
        "n_error": int(counts.get(STATUS_ERROR, 0)),
        "n_scored": int(len(scored)),
        "n_informative_hard": int(hard.fillna(0).sum()) if len(frame) else 0,
        "mean_p_informative": float(scored.mean()) if len(scored) else None,
        "quantiles_p_informative": {
            f"q{int(q * 100):02d}": float(scored.quantile(q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        } if len(scored) else {},
    }


async def classify(args, pending: pd.DataFrame, template: str, backend, prompt_hash: str, fingerprint: str, out_path: Path) -> int:
    client = llm_client.LogprobsClient(
        backend,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_concurrency=args.max_concurrency,
    )
    semaphore = asyncio.Semaphore(client.max_concurrency)
    batch_size = max(1, args.batch_size)
    batches = [pending.iloc[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = [] if args.restart else previous_tables(out_path, prompt_hash, fingerprint, set(pending["tweet_id"]))

    async def _one(text: str) -> dict:
        prompt = text_preprocessing.render_prompt(template, text)
        async with semaphore:
            return await client.classify_prompt(prompt)

    failures = 0
    done = 0
    try:
        for number, batch in enumerate(batches, start=1):
            results = await asyncio.gather(*(_one(t) for t in batch["text"]))
            rows = results_frame(batch, list(results), prompt_hash, fingerprint)
            failures += int((rows["status"] == STATUS_ERROR).sum())
            kept.extend(arrow_table(rows).to_batches())
            write_labels(out_path, kept)
            done += len(rows)
            print(
                f"  batch {number}/{len(batches)}: {done}/{len(pending)} tweets classified",
                flush=True,
            )
    finally:
        await client.aclose()
    return failures


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir = Path(args.output_dir)
    out_path = args.output_dir / LABELS_FILE
    params_path = args.output_dir / PARAMS_FILE

    backend = llm_client.get_backend(args.backend)
    template = text_preprocessing.load_prompt(args.prompt)
    prompt_hash = llm_client.prompt_hash(template)

    if not args.dry_run:
        llm_client.resolve_api_key(backend)  # fail fast with a clear message

    max_distance = None if args.max_distance_km is None else float(args.max_distance_km)
    df = tweet_loader.load_tweets(args.tweets_pkl, cutoff=pd.Timestamp(args.cutoff), max_distance_km=max_distance)
    df = df.sort_values(["tweet_id"], kind="mergesort").reset_index(drop=True)
    # The corpus repeats tweet_ids with identical text and date; scoring them
    # twice would only double the cost. Collapsing here matches what
    # tweet_loader.random_sample does internally, so sampled runs are unchanged.
    n_rows = len(df)
    df = df.drop_duplicates("tweet_id", keep="first").reset_index(drop=True)
    if df.shape[0] != n_rows:
        print(f"Collapsed {n_rows - df.shape[0]} duplicate tweet_id rows: {len(df)} distinct tweets.")
    if args.sample is not None and args.limit is not None:
        raise SystemExit("Use either --sample or --limit, not both.")
    if args.sample is not None:
        df = tweet_loader.random_sample(df, args.sample, args.seed)
        print(
            f"--sample {args.sample} (seed {args.seed}): classifying {len(df)} distinct tweets "
            f"dated {df['date_local'].min()} … {df['date_local'].max()}."
        )
    elif args.limit is not None:
        df = df.head(max(0, int(args.limit)))
        print(f"--limit {args.limit}: classifying {len(df)} tweets.")

    fingerprint = model_fingerprint_json(backend, prompt_hash, args.seed)
    stored = stored_labels(out_path)
    pending = df
    if not args.restart:
        done_ids = labelled_ids(stored, prompt_hash, fingerprint)
        pending = df[~df["tweet_id"].isin(done_ids)]
        if done_ids:
            print(f"Resuming: {len(df) - len(pending)} tweets already labelled by this run.")

    payload = params_payload(args, prompt_hash, backend, len(df), len(pending))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Params -> {params_path}")
    print(f"prompt_hash={prompt_hash[:16]}… model={backend.model}")

    if args.dry_run:
        example = text_preprocessing.render_prompt(template, df["text"].iloc[0]) if len(df) else ""
        print("Dry run: no request sent. Rendered prompt for the first tweet:\n" + "-" * 60)
        print(example)
        return 0

    if pending.empty:
        print("Nothing to do: every selected tweet is already labelled.")
    else:
        asyncio.run(classify(args, pending, template, backend, prompt_hash, fingerprint, out_path))

    ours = stored_labels(out_path)
    ours = ours[(ours["prompt_hash"] == prompt_hash) & (ours["model_fingerprint"] == fingerprint)]
    summary = summarise(ours)
    print("\n--- classification summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  prompt_sha256[:16]: {prompt_hash[:16]}")
    print(f"  output: {out_path}")
    if summary["n_error"]:
        print(f"WARNING: {summary['n_error']} requests failed after {args.max_retries} retries "
              "(status=error). Re-run the script to retry only those tweets.")
    return 1 if summary["n_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
