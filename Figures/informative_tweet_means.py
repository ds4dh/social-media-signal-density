"""Mean number of LLM-informative tweets per day, split by the target label.

Input is the flat score file written by ``signal_density/export_scores.py``
(``signal_density/outputs/tweet_scores.csv``): one row per scored tweet with the
probability that the tweet is informative about emergency demand. Two units of
accounting are reported for every day:

    hard      count of tweets with p_yes >= --threshold (default 0.5, which
              reproduces the ``informative`` column of the file exactly)
    soft      sum of p_yes over the day's tweets, i.e. the fractionally weighted
              count, which carries far less quantisation noise than the hard
              label (see the calibration notes in AGENTS.md)

The daily mean of either quantity is the density lambda of the ceiling model in
``signal_density/bound.py``, so these means are directly comparable to the
lambda the bound was computed with.

The positive/negative split comes from the real labelling code path
(``target_construction``), not a re-implementation: with the defaults
(surge detection, horizon 3, threshold 0.2, respiratory patients) the labels are
label-for-label identical to ``FINAL_DATASET/dataset_matching_horizon_3.pkl``
(checked by ``tests/test_informative_tweet_means.py``). Only days
that carry both a score row and a defined label enter the means, so the two
groups are drawn from the same calendar window.

Outputs (in --output_dir):
    daily_informative_h{H}.csv     per-day table with labels and both counts
    informative_means_h{H}.json    the summary reported on stdout
    fig_informative_by_label_h{H}.png  bar panel + daily time series
    missing_pp_share.csv           side product of the labelling code path

Usage:
    cd Figures && ../.venv/bin/python informative_tweet_means.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
DATASET_CONSTRUCTION_DIR = REPO_ROOT / "Dataset construction"

for _p in (str(REPO_ROOT), str(DATASET_CONSTRUCTION_DIR), str(REPO_ROOT / "Models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SCORES_CSV = REPO_ROOT / "signal_density" / "outputs" / "tweet_scores.csv"
DEFAULT_PROMPT_PARAMS_JSON = (
    REPO_ROOT / "signal_density" / "outputs" / "llm_labels_params.json"
)
#: prompt hash prefix of the full-corpus run (Task.md step 1)
EXPECTED_PROMPT_HASH_PREFIX = "a6224977"

SCORE_COLUMNS = ["tweet_id", "day", "p_yes", "informative", "status", "in_geneva"]
TARGET_COL = "n_target"
CONFIDENCE = 0.95


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores_csv", type=Path, default=DEFAULT_SCORES_CSV,
                        help="Per-tweet LLM score file from export_scores.py.")
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DIR / "outputs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="p_yes above which a tweet counts as informative (hard label).")
    parser.add_argument("--horizon", type=int, default=3,
                        help="Forecast horizon in days for the label.")
    parser.add_argument("--labeling_strategy", default="surge_detection",
                        help="Any target_construction.LABELING_STRATEGIES value.")
    parser.add_argument("--surge_threshold", type=float, default=0.2,
                        help="Relative excess over the 28-day baseline for surge_detection.")
    parser.add_argument("--dispatch_unit", default="patients", choices=("calls", "patients"),
                        help="Target unit of the dispatch-centre series.")
    parser.add_argument("--target_scope", default="respiratory", choices=("total", "respiratory"),
                        help="Case selection of the dispatch-centre series.")
    parser.add_argument("--dataset_pkl", type=Path, default=None,
                        help="Read date/label from an existing FINAL_DATASET pickle instead "
                             "of recomputing the labels from the dispatch exports.")
    parser.add_argument("--geneva_only", action="store_true",
                        help="Restrict the tweet counts to tweets geolocated in the Geneva area.")
    parser.add_argument("--prompt_params_json", type=Path, default=DEFAULT_PROMPT_PARAMS_JSON,
                        help="JSON carrying the prompt_hash of the scoring run (checked only).")
    parser.add_argument("--no_figure", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def check_prompt_hash(path: Path, *, expected: str = EXPECTED_PROMPT_HASH_PREFIX) -> str | None:
    """Return the recorded prompt hash, warning when it is not the full-corpus run."""
    if not Path(path).exists():
        return None
    recorded = str(json.loads(Path(path).read_text(encoding="utf-8")).get("prompt_hash", ""))
    if expected and not recorded.startswith(expected):
        print(
            f"[check_prompt_hash] WARNING: scores come from prompt {recorded[:12]}, "
            f"expected {expected}..."
        )
    return recorded


def load_scores(scores_csv: Path, *, geneva_only: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return the scored tweets of the analysis window as ``day, p_yes, informative``."""
    frame = pd.read_csv(scores_csv, dtype={"tweet_id": str})
    missing = set(SCORE_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError(f"{scores_csv} is missing columns: {sorted(missing)}")

    n_raw = len(frame)
    frame = frame[frame["status"].astype(str) == "ok"].copy()
    n_failed = n_raw - len(frame)
    if n_failed:
        print(f"[load_scores] Dropped {n_failed} tweet(s) whose scoring call did not return ok.")

    frame["day"] = pd.to_datetime(frame["day"]).dt.normalize()
    frame["p_yes"] = frame["p_yes"].astype(float)
    if frame["p_yes"].isna().any():
        raise ValueError(f"{scores_csv} carries {int(frame['p_yes'].isna().sum())} row(s) without a probability.")

    if geneva_only:
        frame = frame[frame["in_geneva"].astype(bool)].copy()
    if verbose:
        print(
            f"[load_scores] {len(frame)} scored tweet(s) over "
            f"{frame['day'].nunique()} day(s) "
            f"({frame['day'].min().date()} -> {frame['day'].max().date()})"
            + (" (Geneva-geolocated only)" if geneva_only else "")
        )
    return frame[["day", "p_yes"]].copy()


def daily_counts(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate per-tweet scores to one row per calendar day.

    Returns ``DataFrame(day, n_tweets, n_informative, soft_sum, max_p)``, where
    ``soft_sum`` is the sum of p_yes and ``n_informative`` the count of tweets
    with ``p_yes >= threshold``.
    """
    flagged = scores.assign(
        informative=scores["p_yes"].ge(threshold).astype(int),
        soft=scores["p_yes"],
    )
    daily = (
        flagged.groupby("day")
        .agg(
            n_tweets=("soft", "size"),
            n_informative=("informative", "sum"),
            soft_sum=("soft", "sum"),
            max_p=("soft", "max"),
        )
        .sort_index()
    )
    return daily.rename_axis("day").reset_index()


def daily_labels(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """Return ``DataFrame(day, n_target, label)`` from the real labelling code path."""
    if args.dataset_pkl is not None:
        ds = pd.read_pickle(args.dataset_pkl)
        labels = ds[["date", "label"]].copy()
        if "n_target" in ds.columns:
            labels["n_target"] = ds["n_target"]
        labels["label"] = labels["label"].astype(float)
        print(f"[daily_labels] {len(labels)} day(s) read from {args.dataset_pkl}.")
    else:
        from types import SimpleNamespace

        from target_construction import labelled_target_series

        params = SimpleNamespace(
            general_dispatch_center_focus=args.dispatch_unit,
            target_scope=args.target_scope,
            labeling_strategy=args.labeling_strategy,
            horizon=args.horizon,
            surge_threshold=args.surge_threshold,
            # keep the reporting-bias side effect inside Figures/
            save_dir=output_dir,
        )
        target = labelled_target_series(params).dropna(subset=["label"])
        labels = target[["date", TARGET_COL, "label"]].copy()
        labels["label"] = labels["label"].astype(float)

    labels = labels.rename(columns={"date": "day"})
    labels["day"] = pd.to_datetime(labels["day"]).dt.normalize()
    return labels.sort_values("day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def poisson_rate_ci(count: float, days: float, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Exact (Garwood) CI for a Poisson rate estimated as ``count / days``."""
    alpha = 1.0 - confidence
    low = 0.0 if count == 0 else stats.chi2.ppf(alpha / 2.0, 2.0 * count) / (2.0 * days)
    high = stats.chi2.ppf(1.0 - alpha / 2.0, 2.0 * count + 2.0) / (2.0 * days)
    return float(low), float(high)


def mean_ci(values: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Two-sided t CI for the mean of the per-day values."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    half = stats.t.ppf(0.5 + confidence / 2.0, n - 1) * stats.sem(values)
    return float(values.mean() - half), float(values.mean() + half)


def rate_ratio_ci(pos_count: float, neg_count: float, pos_days: float, neg_days: float,
                  confidence: float = CONFIDENCE) -> dict:
    """Poisson rate ratio positive/negative with a CI from the conditional binomial.

    Given the total count, the positive share is binomial, so an exact CI on that
    share maps one-to-one onto a CI for the rate ratio. With no informative tweet
    on negative days the ratio is unbounded, and a share that reaches 1 or 0
    carries no information about its width, so those ends stay unbounded.
    """
    nan_ci = {"rate_ratio": float("nan"), "rate_ratio_ci95": (float("nan"), float("nan"))}
    total = int(pos_count + neg_count)
    if total == 0 or pos_days <= 0 or neg_days <= 0:
        return nan_ci

    ratio = (pos_count / pos_days) / (neg_count / neg_days) if neg_count else float("inf")
    share_lo, share_hi = stats.binomtest(int(pos_count), total).proportion_ci(
        confidence_level=confidence, method="exact")
    scale = neg_days / pos_days

    def bound(share):
        return float("inf") if share >= 1.0 else share / (1 - share) * scale

    return {
        "rate_ratio": float(ratio),
        "rate_ratio_ci95": (bound(share_lo), bound(share_hi)),
    }


def group_stats(counts: np.ndarray, n_tweets: np.ndarray) -> dict:
    """Descriptives for one label group: daily counts, Poisson rate, per-tweet rate."""
    days = len(counts)
    total = float(counts.sum())
    out = {
        "n_days": days,
        "n_informative_total": int(total),
        "mean_per_day": float(counts.mean()) if days else float("nan"),
        "std_per_day": float(counts.std(ddof=1)) if days > 1 else float("nan"),
        "median_per_day": float(np.median(counts)) if days else float("nan"),
        "max_per_day": float(counts.max()) if days else float("nan"),
        "mean_ci95": mean_ci(counts),
        "poisson_rate_ci95": poisson_rate_ci(total, days) if days else (float("nan"),) * 2,
        "days_with_at_least_one_share": float(np.mean(counts > 0)) if days else float("nan"),
        "n_tweets_total": int(n_tweets.sum()),
        "mean_tweets_per_day": float(n_tweets.mean()) if days else float("nan"),
    }
    out["informative_per_1000_tweets"] = (
        1000.0 * total / out["n_tweets_total"] if out["n_tweets_total"] else float("nan")
    )
    return out


def summarize(daily: pd.DataFrame, threshold: float) -> dict:
    """Means over all days and over the positive/negative label groups, plus tests."""
    summary: dict = {
        "threshold": threshold,
        "n_days": int(len(daily)),
        "date_min": str(daily["day"].min().date()),
        "date_max": str(daily["day"].max().date()),
        "label_prevalence": float(daily["label"].mean()),
        "overall": group_stats(daily["n_informative"].to_numpy(float), daily["n_tweets"].to_numpy(float)),
        "soft_overall_mean_per_day": float(daily["soft_sum"].mean()),
    }

    groups = {}
    for value, name in ((1.0, "positive"), (0.0, "negative")):
        sub = daily[daily["label"] == value]
        stats_ = group_stats(sub["n_informative"].to_numpy(float), sub["n_tweets"].to_numpy(float))
        stats_["soft_mean_per_day"] = float(sub["soft_sum"].mean()) if len(sub) else float("nan")
        groups[name] = stats_
    summary.update(groups)
    pos, neg = groups["positive"], groups["negative"]
    p_counts = daily.loc[daily["label"] == 1.0, "n_informative"].to_numpy(float)
    n_counts = daily.loc[daily["label"] == 0.0, "n_informative"].to_numpy(float)
    if len(p_counts) > 1 and len(n_counts) > 1:
        welch = stats.ttest_ind(p_counts, n_counts, equal_var=False)
        mann = stats.mannwhitneyu(p_counts, n_counts, alternative="two-sided")
    else:
        welch = mann = None
    summary["comparison"] = {
        "mean_difference_pos_minus_neg": pos["mean_per_day"] - neg["mean_per_day"],
        "mean_difference_ci95": _diff_ci(p_counts, n_counts),
        "welch_t": float(welch.statistic) if welch else float("nan"),
        "welch_p": float(welch.pvalue) if welch else float("nan"),
        "mannwhitney_p": float(mann.pvalue) if mann else float("nan"),
        **rate_ratio_ci(pos["n_informative_total"], neg["n_informative_total"],
                        pos["n_days"], neg["n_days"]),
        "soft_mean_difference": pos["soft_mean_per_day"] - neg["soft_mean_per_day"],
        "soft_rate_ratio": (pos["soft_mean_per_day"] / neg["soft_mean_per_day"]
                            if neg["soft_mean_per_day"] else float("nan")),
    }
    return summary


def _diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch CI for the difference of two means."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    diff = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:  # both groups are constant: the difference is known exactly
        return (diff, diff)
    dof = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    half = stats.t.ppf(0.5 + CONFIDENCE / 2.0, dof) * se
    return (diff - float(half), diff + float(half))


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(daily: pd.DataFrame, summary: dict, path: Path) -> None:
    """Bar panel (mean informative tweets/day by label) + daily time series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, neg = summary["positive"], summary["negative"]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": (1, 2)}
    )

    means = [neg["mean_per_day"], pos["mean_per_day"]]
    lows = [neg["mean_ci95"][0], pos["mean_ci95"][0]]
    highs = [neg["mean_ci95"][1], pos["mean_ci95"][1]]
    # a group can be degenerate (no day, or no variance) -> no error bar
    err = [
        [np.nan_to_num(m - lo) if np.isfinite(lo) else 0.0 for m, lo in zip(means, lows)],
        [np.nan_to_num(hi - m) if np.isfinite(hi) else 0.0 for m, hi in zip(means, highs)],
    ]
    means = [m if np.isfinite(m) else 0.0 for m in means]
    ax1.bar(["negative", "positive"], means, color=["tab:grey", "tab:red"], width=0.55, yerr=err,
            capsize=5, error_kw={"lw": 1.2})
    for x, mean in enumerate(means):
        ax1.text(x, mean, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("informative tweets per day")
    ax1.set_title(
        f"Mean informative tweets/day\nnegative n={neg['n_days']} vs positive n={pos['n_days']}, "
        f"ratio {summary['comparison']['rate_ratio']:.2f}, p={summary['comparison']['welch_p']:.3f}"
    )
    ax1.set_ylim(0, max(max(means), *np.nan_to_num(highs)) * 1.35 or 1.0)
    ax1.grid(axis="y", alpha=0.3)

    pos_days = daily.loc[daily["label"] == 1.0, "day"]
    top = max(daily["n_informative"].max(), daily["soft_sum"].max())
    ax2.plot(daily["day"], daily["n_informative"], lw=0.6, color="tab:red", label="informative tweets/day")
    ax2.plot(daily["day"], daily["soft_sum"], lw=0.6, color="tab:blue", alpha=0.7, label="soft (sum of p_yes)")
    ax2.vlines(pos_days, 0, top, color="tab:orange", alpha=0.15, lw=1.0, label="positive day")
    ax2.set_ylabel("informative tweets")
    ax2.set_title("Daily informative-tweet counts over the study window")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"LLM-informative tweets and the {summary['n_days']}-day labelled window "
                 f"({summary['date_min']} to {summary['date_max']})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _line(label: str, group: dict) -> str:
    lo, hi = group["mean_ci95"]
    plo, phi = group["poisson_rate_ci95"]
    return (
        f"{label:<10} days={group['n_days']:5d}  total={group['n_informative_total']:4d}  "
        f"mean/day={group['mean_per_day']:.4f} [{lo:.4f}, {hi:.4f}]  "
        f"Poisson [{plo:.4f}, {phi:.4f}]  "
        f"median={group['median_per_day']:.2f}  max={group['max_per_day']:.0f}  "
        f"days>=1={group['days_with_at_least_one_share']:.1%}  "
        f"per 1k tweets={group['informative_per_1000_tweets']:.2f}"
    )


def report(summary: dict, scope: str) -> None:
    print(f"\n=== Informative tweets per day ({scope}) ===")
    print(f"window            {summary['date_min']} .. {summary['date_max']}  ({summary['n_days']} days)")
    print(f"hard threshold    p_yes >= {summary['threshold']}")
    print(f"label prevalence  {summary['label_prevalence']:.1%} positive days")
    print()
    print(_line("all days", summary["overall"]))
    print(_line("positive", summary["positive"]))
    print(_line("negative", summary["negative"]))
    comp = summary["comparison"]
    dlo, dhi = comp["mean_difference_ci95"]
    rlo, rhi = comp["rate_ratio_ci95"]
    print(
        f"\ndifference (positive - negative) = {comp['mean_difference_pos_minus_neg']:+.4f}/day "
        f"[{dlo:+.4f}, {dhi:+.4f}], Welch t={comp['welch_t']:.3f} p={comp['welch_p']:.4f}, "
        f"Mann-Whitney p={comp['mannwhitney_p']:.4f}"
    )
    print(
        f"rate ratio (positive / negative) = {comp['rate_ratio']:.3f} [{rlo:.3f}, {rhi:.3f}]"
    )
    print(
        f"soft (sum p_yes) mean/day: positive={summary['positive']['soft_mean_per_day']:.4f}  "
        f"negative={summary['negative']['soft_mean_per_day']:.4f}  "
        f"ratio={comp['soft_rate_ratio']:.3f}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0 < args.threshold <= 1:
        raise SystemExit(f"--threshold must lie in (0, 1], got {args.threshold}.")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    check_prompt_hash(args.prompt_params_json)
    scores = load_scores(args.scores_csv, geneva_only=args.geneva_only)
    daily = daily_counts(scores, args.threshold)
    labels = daily_labels(args, args.output_dir)

    merged = daily.merge(labels, on="day", how="inner", validate="one_to_one")
    merged["label"] = merged["label"].astype(float)
    dropped = len(daily) - len(merged)
    if dropped:
        print(f"[main] {dropped} scored day(s) had no defined label and were dropped.")
    merged = merged.sort_values("day").reset_index(drop=True)

    summary = summarize(merged, args.threshold)
    scope = "Geneva-geolocated tweets" if args.geneva_only else "all scored tweets"
    report(summary, scope)

    stem = f"{'informative_means' if not args.geneva_only else 'informative_means_geneva'}_h{args.horizon}"
    daily_path = args.output_dir / f"daily_informative_h{args.horizon}{'_geneva' if args.geneva_only else ''}.csv"
    out = merged.copy()
    out["day"] = out["day"].dt.strftime("%Y-%m-%d")
    out.to_csv(daily_path, index=False)

    params = {
        "script": "Figures/informative_tweet_means.py",
        "scores_csv": str(args.scores_csv),
        "threshold": args.threshold,
        "horizon": args.horizon,
        "labeling_strategy": args.labeling_strategy,
        "surge_threshold": args.surge_threshold,
        "dispatch_unit": args.dispatch_unit,
        "target_scope": args.target_scope,
        "dataset_pkl": str(args.dataset_pkl) if args.dataset_pkl else None,
        "geneva_only": args.geneva_only,
        "confidence": CONFIDENCE,
    }
    payload = {"parameters": params, **summary}
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(f"\nDaily table -> {daily_path}")
    print(f"Summary       -> {json_path}")

    if not args.no_figure:
        fig_path = args.output_dir / f"fig_informative_by_label_h{args.horizon}{'_geneva' if args.geneva_only else ''}.png"
        make_figure(merged, summary, fig_path)
        print(f"Figure        -> {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
