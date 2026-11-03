"""AUROC of every tweet model per injection rate lambda0, against the dispatch-only baseline.

Input is one output directory of `Experiments/semi_synthetic/run_experiment.py`
(``outputs/<experiment>/``, the default is the BERT-PCA run without ``n_tweets``):

    results.csv      one row per (model, lambda0, seed, fold) with its test AUROC
    predictions.csv  per-day test predictions of every model (optional, enables pooled AUROC)
    config.json      run configuration (used to echo the experiment settings)

The baseline is the ``model == "baseline"`` row set: XGBoost on dispatch history
only, no tweet column. It is trained once per fold and shared by every lambda,
so gains are paired by fold.

Two AUROC units are reported for each lambda, because they answer different
questions and the folds of this experiment are strongly heterogeneous
(per-fold baseline AUROC 0.561/0.516/0.500):

    mean_of_folds  average of the per-fold AUROC, averaging over seeds first.
                   Directly comparable to the baseline's mean over folds and to
                   the ``gain`` column already stored in results.csv.
    pooled         single AUROC over all test rows of all folds and seeds. Labels
                   are identical across seeds for a given (fold, date), which is
                   checked before pooling; pooling therefore mixes the three
                   test windows and is dominated by prevalence differences
                   between folds rather than by within-window discrimination.

Uncertainty is the spread across injection seeds, not a sampling CI: lambda is
fixed by the design, so the only stochasticity is which tweets were drawn.

Outputs (in --output_dir):
    model_performance_<experiment>.csv   one row per (model, variant, lambda0)
    model_performance_<experiment>.json  same table + baseline + run config
    fig_model_performance_<experiment>.png  AUROC and gain vs lambda

Usage:
    cd Figures && ../.venv/bin/python model_performance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
EXPERIMENT_OUTPUTS_DIR = REPO_ROOT / "Experiments" / "semi_synthetic" / "outputs"
DEFAULT_EXPERIMENT = "bert_pca_75_horizon_3_without_n_tweets"

#: value of results.csv["model"] holding the no-tweet reference model
BASELINE_MODEL = "baseline"
#: columns identifying one experimental variant (a model on one pool/background pair)
VARIANT_COLUMNS = ["model", "pool", "background"]
LAMBDA_COLUMN = "lambda0"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experiment_dir", type=Path,
                        default=EXPERIMENT_OUTPUTS_DIR / DEFAULT_EXPERIMENT,
                        help="Directory written by run_experiment.py (must hold results.csv).")
    parser.add_argument("--results_csv", type=Path, default=None,
                        help="Override the path of results.csv (default: <experiment_dir>/results.csv).")
    parser.add_argument("--predictions_csv", type=Path, default=None,
                        help="Override the path of predictions.csv "
                             "(default: <experiment_dir>/predictions.csv, used only when present).")
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DIR / "outputs")
    parser.add_argument("--model", action="append", default=None,
                        help="Restrict to this tweet model (repeatable). Default: every non-baseline model.")
    parser.add_argument("--pool", default=None,
                        help="Restrict to one pool value (e.g. informative). "
                             "Default: automatic when the run has a single pool.")
    parser.add_argument("--background", default=None,
                        help="Restrict to one background variant. Default: automatic when unique.")
    parser.add_argument("--no_figure", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_results(path: Path) -> pd.DataFrame:
    """Read results.csv and normalise the key columns."""
    frame = pd.read_csv(path)
    missing = {"model", "fold", "auroc", LAMBDA_COLUMN} - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    for col in VARIANT_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["fold"] = frame["fold"].astype(int)
    frame["auroc"] = frame["auroc"].astype(float)
    frame[LAMBDA_COLUMN] = pd.to_numeric(frame[LAMBDA_COLUMN], errors="coerce")
    return frame


def load_predictions(path: Path) -> pd.DataFrame:
    """Read predictions.csv, returning an empty frame when it does not exist."""
    if not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"])
    frame[LAMBDA_COLUMN] = pd.to_numeric(frame[LAMBDA_COLUMN], errors="coerce")
    frame["fold"] = frame["fold"].astype(int)
    return frame


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, pool, background) of the tweet models, with its lambda grid."""
    tweet_rows = results[results["model"] != BASELINE_MODEL]
    rows = []
    for keys, sub in tweet_rows.groupby(VARIANT_COLUMNS, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({
            **dict(zip(VARIANT_COLUMNS, keys)),
            "n_lambdas": sub[LAMBDA_COLUMN].nunique(),
            "n_seeds": int(sub["seed"].nunique()),
            "lambdas": sorted(sub[LAMBDA_COLUMN].dropna().unique().tolist()),
        })
    return pd.DataFrame(rows)


def select_variants(frame: pd.DataFrame, *, model=None, pool=None, background=None) -> pd.DataFrame:
    """Filter the variant table; error out when the filter selects nothing."""
    out = frame
    for column, value in (("model", model), ("pool", pool), ("background", background)):
        if value is None:
            continue
        wanted = [value] if isinstance(value, str) else list(value)
        out = out[out[column].isin(wanted)]
    if out.empty:
        raise SystemExit(
            "No experiment variant matches --model/--pool/--background. Available variants:\n"
            + frame.drop(columns=["lambdas"]).to_string(index=False)
        )
    return out


def auto_variant_filter(variants_frame: pd.DataFrame, args: argparse.Namespace) -> argparse.Namespace:
    """Pin pool/background when the run has a single value, so rows are never mixed."""
    for column in ("pool", "background"):
        if getattr(args, column) is not None:
            continue
        values = variants_frame[column].dropna().unique().tolist()
        if len(values) > 1:
            raise SystemExit(
                f"results.csv holds several values of '{column}' ({values}); pass --{column} "
                "so that variants are not averaged together."
            )
        setattr(args, column, values[0] if values else None)
    return args


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def baseline_summary(results: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Per-fold AUROC of the dispatch-only model, plus mean over folds and pooled."""
    base = results[results["model"] == BASELINE_MODEL]
    if base.empty:
        raise SystemExit("results.csv holds no rows with model == 'baseline'; cannot compute gains.")
    per_fold = base.groupby("fold")["auroc"].mean().sort_index()
    out = {
        "n_baseline_rows": int(len(base)),
        "auroc_by_fold": {int(f): float(a) for f, a in per_fold.items()},
        "auroc_mean_folds": float(per_fold.mean()),
        "auroc_min_fold": float(per_fold.min()),
        "auroc_max_fold": float(per_fold.max()),
        "auroc_pooled": float("nan"),
    }
    preds = _variant_predictions(predictions, BASELINE_MODEL)
    if len(preds):
        out["auroc_pooled"] = float(roc_auc_score(preds["label"], preds["proba"]))
    return out


def _variant_predictions(predictions: pd.DataFrame, model: str, *, pool=None,
                         background=None) -> pd.DataFrame:
    """Rows of predictions.csv belonging to one variant (baseline ignores pool/background)."""
    if predictions.empty:
        return predictions
    sel = predictions["model"] == model
    if model != BASELINE_MODEL:
        if pool is not None:
            sel &= predictions["pool"].astype(str).eq(str(pool))
        if background is not None:
            sel &= predictions["background"].astype(str).eq(str(background))
    return predictions[sel].copy()


# ---------------------------------------------------------------------------
# per-lambda aggregation
# ---------------------------------------------------------------------------
def check_labels_agree(predictions: pd.DataFrame) -> None:
    """Pooled AUROC is only comparable to the baseline if both saw the same labels."""
    if predictions.empty:
        return
    base = _variant_predictions(predictions, BASELINE_MODEL)[["fold", "date", "label"]]
    if base.empty:
        return
    tweet = predictions[predictions["model"] != BASELINE_MODEL][["fold", "date", "label"]].drop_duplicates()
    merged = tweet.merge(base, on=["fold", "date"], how="inner", suffixes=("_model", "_baseline"))
    disagree = int((merged["label_model"].astype(float) != merged["label_baseline"].astype(float)).sum())
    if disagree:
        raise SystemExit(
            f"predictions.csv: {disagree} test row(s) have a label that differs between the "
            "tweet model and the baseline — pooled AUROC would not be comparable."
        )


def per_lambda_frame(results: pd.DataFrame, predictions: pd.DataFrame, base: dict,
                     *, pool=None, background=None) -> pd.DataFrame:
    """Summarise each (model, lambda0) of one variant against the shared baseline."""
    tweet_rows = results[results["model"] != BASELINE_MODEL].copy()
    if pool is not None:
        tweet_rows = tweet_rows[tweet_rows["pool"].astype(str).eq(str(pool))]
    if background is not None:
        tweet_rows = tweet_rows[tweet_rows["background"].astype(str).eq(str(background))]
    if tweet_rows.empty:
        raise SystemExit("results.csv holds no tweet-model rows after the pool/background filter.")

    rows = []
    for (model, lam), sub in tweet_rows.groupby(["model", LAMBDA_COLUMN], sort=True):
        per_fold_gain = sub.groupby("fold")["gain"].mean()
        per_fold_auroc = sub.groupby("fold")["auroc"].mean()
        per_seed = sub.groupby("seed")["auroc"].mean()
        row = {
            "model": model, "pool": sub["pool"].iloc[0], "background": sub["background"].iloc[0],
            LAMBDA_COLUMN: lam,
            "n_seeds": int(sub["seed"].nunique()), "n_folds": int(sub["fold"].nunique()),
            "auroc_mean_folds": float(per_fold_auroc.mean()),
            "auroc_seed_min": float(per_seed.min()), "auroc_seed_max": float(per_seed.max()),
            "baseline_auroc_mean_folds": float(np.mean([base["auroc_by_fold"][f] for f in per_fold_auroc.index])),
            "gain_mean_folds": float(per_fold_gain.mean()),
            "gain_seed_min": float(per_seed.min() - base["auroc_mean_folds"]),
            "gain_seed_max": float(per_seed.max() - base["auroc_mean_folds"]),
            "gain_min_fold": float(per_fold_gain.min()), "gain_max_fold": float(per_fold_gain.max()),
            "auroc_pooled": float("nan"), "gain_pooled": float("nan"), "n_test_pooled": 0,
            # n_injected_total is constant across folds within a (lambda, seed) draw,
            # so the mean over all rows is the mean injected count per seed.
            "n_injected_per_seed": float(sub["n_injected_total"].mean()),
            "n_features_min": int(sub["n_features"].min()), "n_features_max": int(sub["n_features"].max()),
        }
        row.update(_pooled_metrics(predictions, model, lam, base, pool=pool, background=background))
        rows.append(row)
    return pd.DataFrame(rows)


def _pooled_metrics(predictions: pd.DataFrame, model: str, lam, base: dict, *, pool, background) -> dict:
    """AUROC over all test rows of all folds and seeds at this lambda."""
    empty = {"auroc_pooled": float("nan"), "gain_pooled": float("nan"), "n_test_pooled": 0}
    preds = _variant_predictions(predictions, model, pool=pool, background=background)
    if preds.empty:
        return empty
    preds = preds[np.isclose(pd.to_numeric(preds[LAMBDA_COLUMN], errors="coerce"), float(lam))]
    if preds.empty or preds["label"].nunique() < 2:
        return empty
    auroc = float(roc_auc_score(preds["label"], preds["proba"]))
    return {
        "auroc_pooled": auroc,
        "gain_pooled": auroc - base["auroc_pooled"],
        "n_test_pooled": int(len(preds)),
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(summary: pd.DataFrame, base: dict, path: Path) -> None:
    """AUROC versus lambda with the baseline line, plus the gain panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = plt.get_cmap("tab10")
    models = summary["model"].drop_duplicates().tolist()
    ax2.axhline(0.0, color="black", lw=0.8)

    for i, model in enumerate(models):
        sub = summary[summary["model"] == model].sort_values(LAMBDA_COLUMN)
        xs = sub[LAMBDA_COLUMN].to_numpy(float)
        ys = sub["auroc_mean_folds"].to_numpy(float)
        ax1.plot(xs, ys, marker="o", ms=4, color=colors(i), label=model)
        ax1.fill_between(xs, sub["auroc_seed_min"], sub["auroc_seed_max"], color=colors(i), alpha=0.18, lw=0)
        ax2.plot(xs, sub["gain_mean_folds"].to_numpy(float), marker="o", ms=4, color=colors(i), label=model)
        ax2.fill_between(xs, sub["gain_seed_min"], sub["gain_seed_max"], color=colors(i), alpha=0.18, lw=0)

    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"injected informative tweets per day $\lambda_0$")
        ax.grid(alpha=0.3)
    ax1.axhline(base["auroc_mean_folds"], color="black", ls="--", lw=1.2,
                label=f"baseline {base['auroc_mean_folds']:.3f}")
    ax1.set_ylabel("test AUROC (mean of folds)")
    ax1.set_title("AUROC per injection rate")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.set_ylabel(r"AUROC gain over baseline $\Delta$")
    ax2.set_title("Gain over the dispatch-only baseline")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Semi-synthetic injection: BERT-PCA model vs dispatch-only baseline "
                 "(shaded band = spread over injection seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def report(summary: pd.DataFrame, base: dict, cfg: dict, scope: str) -> None:
    print(f"\n=== Model performance vs baseline ({scope}) ===")
    if cfg:
        print(f"experiment   {cfg.get('experiment', '?')}  |  method={cfg.get('method', '?')}"
              f"  dataset={cfg.get('dataset_name', '?')}  pca_threshold={cfg.get('pca_threshold', '?')}"
              f"  num_trials={cfg.get('num_trials', '?')}")
    folds = " ".join(f"f{k}={v:.4f}" for k, v in sorted(base["auroc_by_fold"].items()))
    print("BASELINE (dispatch history only, no tweets)")
    print(f"  mean of folds = {base['auroc_mean_folds']:.4f}   {folds}")
    if np.isfinite(base["auroc_pooled"]):
        print(f"  pooled        = {base['auroc_pooled']:.4f}")
    print()
    header = (f"{'lambda0':>8} {'n_inj':>7} {'seeds':>5} {'AUROC':>8} {'seed range':>17} "
              f"{'gain':>8} {'gain seed range':>19} {'pooled':>8} {'gain pool':>10}")
    print(header)
    print("-" * len(header))
    for row in summary.itertuples(index=False):
        seed_range = f"[{row.auroc_seed_min:.4f}, {row.auroc_seed_max:.4f}]"
        gain_range = f"[{row.gain_seed_min:+.4f}, {row.gain_seed_max:+.4f}]"
        pooled = f"{row.auroc_pooled:8.4f}" if np.isfinite(row.auroc_pooled) else f"{'n/a':>8}"
        gain_pool = f"{row.gain_pooled:+10.4f}" if np.isfinite(row.gain_pooled) else f"{'n/a':>10}"
        print(f"{getattr(row, 'lambda0'):8.1f} {row.n_injected_per_seed:7.0f} {row.n_seeds:5d} "
              f"{row.auroc_mean_folds:8.4f} {seed_range:>17} {row.gain_mean_folds:+8.4f} "
              f"{gain_range:>19} {pooled} {gain_pool}")
    print("\nn_inj = mean number of tweets injected per seed over the whole reference window.")
    best = summary.loc[summary["gain_mean_folds"].idxmax()]
    print(
        f"\nBest mean-of-folds gain: lambda0={getattr(best, 'lambda0'):g} "
        f"({best.model}, {best.pool}/{best.background}) at {best.gain_mean_folds:+.4f} "
        f"-> AUROC {best.auroc_mean_folds:.4f} vs baseline {base['auroc_mean_folds']:.4f}."
    )
    positive = summary[summary["gain_mean_folds"] > 0][LAMBDA_COLUMN].tolist()
    print("Lambda values beating the baseline (mean of folds): "
          + (", ".join(f"{v:g}" for v in positive) if positive else "none"))


def main(argv=None) -> int:
    args = parse_args(argv)
    results_csv = args.results_csv or args.experiment_dir / "results.csv"
    predictions_csv = args.predictions_csv or args.experiment_dir / "predictions.csv"
    if not results_csv.exists():
        raise SystemExit(f"{results_csv} does not exist — pass --experiment_dir or --results_csv.")
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.experiment_dir / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    experiment = cfg.get("experiment", args.experiment_dir.name)

    results = load_results(results_csv)
    predictions = load_predictions(predictions_csv)
    if predictions.empty:
        print(f"[main] {predictions_csv} not found: pooled AUROC is skipped, gains come from results.csv.")
    check_labels_agree(predictions)

    variant_table = variants(results)
    if variant_table.empty:
        raise SystemExit("results.csv holds no tweet-model rows — nothing to compare against the baseline.")
    print("[variants] tweet models found in results.csv:")
    print(variant_table.drop(columns=["lambdas"]).to_string(index=False))
    variant_table = select_variants(variant_table, model=args.model, pool=args.pool,
                                    background=args.background)
    args = auto_variant_filter(variant_table, args)

    base = baseline_summary(results, predictions)
    summary = per_lambda_frame(results, predictions, base, pool=args.pool, background=args.background)
    if len(variant_table) < len(variants(results)):
        keep = variant_table[["model"]].drop_duplicates()
        summary = summary.merge(keep, on="model", how="inner")

    report(summary, base, cfg, scope=f"pool={args.pool}, background={args.background}")

    stem = f"model_performance_{experiment}"
    csv_path = args.output_dir / f"{stem}.csv"
    summary.to_csv(csv_path, index=False)
    payload = {
        "script": "Figures/model_performance.py",
        "results_csv": str(results_csv),
        "predictions_csv": str(predictions_csv) if predictions.size else None,
        "config": cfg,
        "baseline": base,
        "per_lambda": summary.to_dict(orient="records"),
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    print(f"\nTable  -> {csv_path}")
    print(f"Summary -> {json_path}")

    if not args.no_figure:
        fig_path = args.output_dir / f"fig_{stem}.png"
        make_figure(summary, base, fig_path)
        print(f"Figure -> {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
