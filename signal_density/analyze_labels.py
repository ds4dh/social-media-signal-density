"""Summary metrics for a full LLM classification run.

Reports the volume of ground-level signal the labels imply: hard-positive rate
with a binomial CI, the implied daily call-adjacent tweet rate (lambda) at
several score thresholds, score-distribution statistics, and the split by
language, year and calendar day.

Usage:
    ../.venv/bin/python analyze_labels.py --input outputs_full/llm_labels.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR.parent))

from signal_density import llm_client, text_preprocessing  # noqa: E402

THRESHOLDS = (0.5, 0.3, 0.1, 0.05)


def poisson_ci(count: int, exposure: float, level: float = 0.95) -> tuple[float, float]:
    """Garwood interval for a Poisson rate ``count / exposure``."""
    from scipy.stats import chi2

    if exposure <= 0:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    lo = 0.0 if count == 0 else chi2.ppf(alpha / 2, 2 * count) / 2.0
    hi = chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2.0
    return (lo / exposure, hi / exposure)


def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - level) / 2)
    ph = k / n
    den = 1 + z**2 / n
    center = (ph + z**2 / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def entropy(p: pd.Series) -> float:
    return float(-(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12)).mean())


def load_labels(input_path: Path, prompt: str | None) -> pd.DataFrame:
    frame = pd.read_parquet(input_path)
    frame["tweet_id"] = frame["tweet_id"].astype(str)
    if prompt:
        wanted = llm_client.prompt_hash(text_preprocessing.load_prompt(prompt))
        frame = frame[frame["prompt_hash"] == wanted]
    elif frame["prompt_hash"].nunique() > 1:
        counts = frame.groupby("prompt_hash").size()
        frame = frame[frame["prompt_hash"] == counts.idxmax()]
        print(
            f"warning: {input_path} holds {counts.size} prompt versions; "
            f"using the largest block ({counts.idxmax()[:12]}, {counts.max()} rows). "
            "Pass --prompt to be explicit."
        )
    return frame.reset_index(drop=True)


def metrics(frame: pd.DataFrame, tweets_per_day: float | None = None) -> dict:
    """Summary stats. ``tweets_per_day`` rescales a subsample to the corpus.

    The Poisson exposure is the inclusive calendar span of the labelled tweets.
    For a full-corpus run the observed tweets-per-day is the right exposure and
    lambda is just positives/day; for a subsample it must be the corpus volume,
    otherwise lambda would be diluted by the sampling fraction.
    """
    ok = frame["status"].astype(str).eq("ok")
    scored = frame[ok].copy()
    p = pd.to_numeric(scored["p_informative"], errors="coerce").astype(float)
    scored = scored.assign(p=p).dropna(subset=["p"])
    p = scored["p"]
    days = pd.to_datetime(scored["date_local"]).dt.normalize()
    span_days = int((days.max() - days.min()).days) + 1
    observed_per_day = len(p) / span_days if span_days else 0.0
    scale = (tweets_per_day or observed_per_day) / observed_per_day if observed_per_day else 1.0
    n_days = span_days
    out: dict = {
        "n_rows": int(len(frame)),
        "n_ok": int(ok.sum()),
        "n_error": int(frame["status"].astype(str).eq("error").sum()),
        "n_no_answer": int(frame["status"].astype(str).eq("no_answer").sum()),
        "n_scored": int(len(scored)),
        "date_min": str(days.min().date()),
        "date_max": str(days.max().date()),
        "n_days": n_days,
    }
    hard = p >= 0.5
    k = int(hard.sum()) * scale
    lo, hi = wilson_ci(int(hard.sum()), len(p))
    lam_lo, lam_hi = (b * scale for b in poisson_ci(int(hard.sum()), n_days))
    out.update(
        {
            "n_positive_hard": int(hard.sum()),
            "scaled_positive_hard": k,
            "rate_hard": float(hard.mean()) if len(p) else None,
            "rate_hard_ci95": [lo, hi],
            "lambda_hard_per_day": k / n_days if n_days else None,
            "lambda_hard_ci95": [lam_lo, lam_hi],
            "mean_p": float(p.mean()),
            "median_p": float(p.median()),
            "q75_p": float(p.quantile(0.75)),
            "q90_p": float(p.quantile(0.90)),
            "q99_p": float(p.quantile(0.99)),
            "max_p": float(p.max()),
            "share_p_below_0.01": float((p < 0.01).mean()),
            "share_p_below_0.05": float((p < 0.05).mean()),
            "entropy_mean": entropy(p),
            "lambda_soft_per_day": float(p.sum() * scale / n_days) if n_days else None,
            "tweets_per_day": tweets_per_day or observed_per_day,
        }
    )
    thresholds = {}
    for thr in THRESHOLDS:
        c = int((p >= thr).sum())
        cs = c * scale
        tlo, thi = (b * scale for b in poisson_ci(c, n_days))
        thresholds[str(thr)] = {
            "n": c,
            "share": c / len(p) if len(p) else None,
            "lambda_per_day": cs / n_days if n_days else None,
            "lambda_ci95": [tlo, thi],
        }
    out["thresholds"] = thresholds
    return out


def group_rates(scored: pd.DataFrame, key: str) -> pd.DataFrame:
    g = scored.assign(positive=scored["p"] >= 0.5).groupby(key, observed=True)
    table = g.agg(n=("p", "size"), positives=("positive", "sum"), mean_p=("p", "mean"), max_p=("p", "max"))
    table["rate"] = table["positives"] / table["n"]
    return table.sort_values("rate", ascending=False)


def daily_counts(scored: pd.DataFrame) -> pd.DataFrame:
    day = pd.to_datetime(scored["date_local"]).dt.normalize()
    tab = (
        scored.assign(day=day, positive=scored["p"] >= 0.5, soft=scored["p"])
        .groupby("day")
        .agg(n_tweets=("p", "size"), positives=("positive", "sum"), soft_sum=("soft", "sum"))
    )
    tab["rate"] = tab["positives"] / tab["n_tweets"]
    return tab


def print_metrics(res: dict) -> None:
    print("\n--- volume ---")
    for key in ("n_rows", "n_ok", "n_error", "n_no_answer", "n_scored", "date_min", "date_max", "n_days"):
        print(f"  {key}: {res[key]}")
    print("\n--- hard positives (p >= 0.5) ---")
    lo, hi = res["rate_hard_ci95"]
    llo, lhi = res["lambda_hard_ci95"]
    print(f"  n_positive_hard: {res['n_positive_hard']}  (corpus-scaled: {res['scaled_positive_hard']:.0f})")
    print(f"  rate: {res['rate_hard']:.4f}  [95% CI {lo:.4f} .. {hi:.4f}]")
    print(f"  lambda (tweets/day): {res['lambda_hard_per_day']:.2f}  [95% CI {llo:.2f} .. {lhi:.2f}]")
    print("\n--- score distribution ---")
    for key in ("mean_p", "median_p", "q75_p", "q90_p", "q99_p", "max_p",
                "share_p_below_0.01", "share_p_below_0.05", "entropy_mean", "lambda_soft_per_day"):
        print(f"  {key}: {res[key]:.4f}")
    print("\n--- lambda by threshold ---")
    for thr, vals in res["thresholds"].items():
        t = "[%.2f .. %.2f]" % tuple(vals["lambda_ci95"])
        print(f"  p>={thr}: n={vals['n']:6d}  share={vals['share']:.4f}  lambda={vals['lambda_per_day']:7.2f}/day {t}")
    print(f"  soft (sum p):        lambda={res['lambda_soft_per_day']:7.2f}/day")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=Path("signal_density/outputs/llm_labels.parquet"))
    parser.add_argument("--prompt", default=None, help="Prompt name to filter by (default: the largest block).")
    parser.add_argument("--csv_out", type=Path, default=None, help="Write the per-day table here.")
    parser.add_argument("--json_out", type=Path, default=None, help="Write the summary dict here.")
    parser.add_argument("--top", type=int, default=15, help="How many rows of each group table to print.")
    parser.add_argument("--tweets_per_day", type=float, default=None,
                        help="Corpus tweet volume per day, to rescale a subsample to corpus lambda. "
                             "Default: the observed rate in this file (correct for a full run).")
    args = parser.parse_args(argv)

    frame = load_labels(args.input, args.prompt)
    if frame.empty:
        raise SystemExit(f"No rows for prompt {args.prompt!r} in {args.input}.")
    print(f"Labels file: {args.input}  prompt_hash={frame['prompt_hash'].iloc[0][:12]}  rows={len(frame)}")

    scored = frame[frame["status"].astype(str).eq("ok")].copy()
    scored["p"] = pd.to_numeric(scored["p_informative"], errors="coerce")
    scored = scored.dropna(subset=["p"])

    res = metrics(frame, args.tweets_per_day)
    print_metrics(res)

    print(f"\n--- language (top {args.top} of {scored['lang'].nunique()} by positive rate) ---")
    lang = group_rates(scored, "lang")
    print(lang.head(args.top).round(4).to_string())

    print("\n--- year ---")
    yearly = group_rates(scored.assign(year=pd.to_datetime(scored["date_local"]).dt.year), "year")
    print(yearly.sort_index().round(4).to_string())

    print("\n--- geolocation ---")
    geo = group_rates(scored, "in_geneva")
    print(geo.round(4).to_string())

    daily = daily_counts(scored)
    pos_days = int((daily["positives"] > 0).sum())
    conc = daily["positives"].sort_values(ascending=False)
    print("\n--- temporal concentration ---")
    print(f"  days with >=1 positive: {pos_days}/{res['n_days']} ({pos_days / res['n_days']:.1%})")
    top10 = int(conc.head(10).sum()); total = int(conc.sum())
    print(f"  share of positives in the top 10 days: {top10}/{total} ({top10 / total:.1%})" if total else "  no positives")
    print(f"  daily positives: mean={daily['positives'].mean():.2f} sd={daily['positives'].std():.2f} max={int(conc.iloc[0]) if total else 0}")
    print("  top days by positives:")
    print(conc.head(args.top).astype(int).to_string())

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.csv_out)
        print(f"\nPer-day table -> {args.csv_out}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        print(f"Summary -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
