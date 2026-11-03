r"""Semi-synthetic experiment: can an injected tweet signal predict a surge on its own?

The tweet model sees **tweet features only** — no dispatch history, no calendar
features (see FEATURE_SETS). Performance is the absolute test AUROC; no gain over
a baseline is computed by default.

One experiment =
1. For each lambda0 (and seed): inject k_t ~ Poisson(rate_t) relevant pool
   tweets per day into the real background and rebuild the daily tweet features on
   top of the reference dataset given by --dataset_name. The default
   --injection_mode 'window' puts a surge day's elevated rate on the
   --feature_window days that end --horizon days before the surge (surge at t
   raises t-h, t-h-1, t-h-w+1), capped at
   lambda0 * (1 + delta) so overlapping surges never stack; 'step' keeps the
   elevated count on the surge day and 'ratio' scales it with r_t. See inject.py.
   Every day draws from the single pool spanning the whole period, so the same
   tweet can appear in both the training and the test days of one fold.
2. For each synthetic dataset: tune the --estimator model (XGBoost by default,
   logistic regression or LightGBM) per fold with Optuna on validation,
   refit with the best parameters, evaluate test AUROC.
3. Print the per-fold test AUROC and the fold-averaged AUROC.
4. Append every result to CSV files for later analysis.

The dispatch-history-only reference model is not trained by default. Pass
``--with_baseline`` to train it and fill the ``auroc_baseline``/``gain`` columns
of results.csv; they stay NaN otherwise, and Figures/model_performance.py (which
reports gains) then has nothing to read.

Outputs, in outputs/<experiment>/:
    config.json         resolved parameters of the run
    results.csv         one row per (model, lambda0, seed, fold)
    predictions.csv     per-day test predictions (date, label, proba) of every model
    pool_report.json      size and date span of the single injection pool
    datasets/           synthetic datasets, only with --save_datasets

Resumable: (lambda0, seed) combinations already complete in results.csv are skipped,
and a baseline already in results.csv is reused.

Run from Experiments/semi_synthetic/:
    python run_experiment.py --experiment lexicon_v1 --method lexicon \
        --dataset_name dataset_matching_surge_patients_horizon_5.pkl \
        --lambdas 0 0.1 1 10 100 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import paths  # noqa: F401  (puts Dataset construction/ and Models/ on sys.path)
from constant import N_JOBS
from constants import NLP_FEATURE_LAGS
from corpus import build_background, build_pool, load_llm_labels, load_unique_tweets, reference_days
from dataset import construct_cv_datasets, select_features
from features import build_daily_features, fit_bert_reducer
from inject import (
    INJECTION_MODES,
    POOL_REPORT_FILE,
    STEP_MODE,
    WINDOW_MODE,
    inject,
    pool_report,
    surge_ratio,
    surge_window_coverage,
)
# Bare names, like every other import here: Models/ is on sys.path via paths, and
# `Models.OurTree` would load a *second* copy of the module (Models/ has no
# __init__.py, so it is a namespace package). OurLinear.py inherits the bare-name
# OurTreeModel, so a prefixed import here would give two distinct classes and
# every issubclass/isinstance relation between them would silently be False.
from OurLightGBM import OurLightGBMModel
from OurLinear import OurLinearModel
from OurTree import OurTreeModel
from paths import FEATURE_WINDOW, HORIZON, OUTPUTS_DIR, REFERENCE_DATASET_PKL, SURGE_THRESHOLD

DATASET_DIR = REFERENCE_DATASET_PKL.parent

#: seed of the Optuna sampler and of every XGBoost fit (Models/main.py default).
#: Independent of the injection seed.
MODEL_SEED = 15

#: model -> (feature_set, OurTreeModel approach)
#: The tweet models get the tweet columns *only*: no 'patients' (dispatch history
#: of the target series) and no 'time' (calendar features), so their AUROC measures
#: how much of the surge label the tweet channel carries by itself. The 'baseline'
#: entry is the reference model and is only trained under --with_baseline.
FEATURE_SETS = {
    "baseline": (["patients", "time"], "matching"),
    "lexicon": (["matching"], "matching"),
    "bert": (["bert"], "BERT"),
}

#: ``--estimator`` -> model class. All classes share the Optuna search, the refit
#: with the best parameters and the evaluation of this module; only the estimator
#: family (Models/OurTree.py vs Models/OurLinear.py vs Models/OurLightGBM.py)
#: changes.
ESTIMATORS = {"xgboost": OurTreeModel, "logistic": OurLinearModel,
              "lightgbm": OurLightGBMModel}

RESULT_COLUMNS = [
    "experiment", "dataset", "model", "pool", "background", "lambda0", "seed", "fold",
    "auroc", "auroc_baseline", "gain", "val_auroc", "n_injected_total",
    "n_features", "n_test", "n_pos_test", "num_trials", "seconds", "best_params",
]
KEY_COLUMNS = ["model", "pool", "background", "lambda0", "seed"]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def make_args(approach: str, num_trials: int, pca_threshold: float,
              skip_fold_pca: bool = False) -> types.SimpleNamespace:
    """Namespace expected by Models/dataset.py and OurTreeModel. Every value explicit."""
    return types.SimpleNamespace(
        approach=approach,
        random_seed=MODEL_SEED,
        n_jobs=N_JOBS,
        num_trials=num_trials,
        labeling_strategy="surge_detection",
        horizon=HORIZON,
        dataset_split_strategy="time_based",
        train_set_proportion=0.7,
        validation_set_proportion=0.15,
        test_set_proportion=0.15,
        pca_variance_threshold=pca_threshold,
        skip_fold_pca=skip_fold_pca,
    )


def tune_and_evaluate(frame: pd.DataFrame, model: str, cfg, logdir: Path) -> list[dict]:
    """Per fold: Optuna search on validation, refit with the best params, test AUROC."""
    feature_set, approach = FEATURE_SETS[model]
    # With the full-cache PCA the embeddings arrive pre-reduced, so the per-fold
    # reduction of Models/dataset.py must not run on top of it.
    args = make_args(approach, cfg.num_trials, cfg.pca_threshold,
                     skip_fold_pca=getattr(cfg, "bert_pca_reducer", None) is not None)
    folds = construct_cv_datasets(select_features(frame, feature_set), args=args)
    logdir.mkdir(parents=True, exist_ok=True)

    model_cls = ESTIMATORS[getattr(cfg, "estimator", "xgboost")]
    out = []
    for fold in folds:
        t0 = time.time()
        tree = model_cls(
            param=args,
            dataset={k: fold[k] for k in ("train", "validation", "test")},
            logdir=str(logdir),
        )
        study = tree._hyperparameter_search()
        tree.model = tree._construct_model(study.best_params)
        tree.model.fit(tree.X_train, tree.y_train,
                       eval_set=[(tree.X_val, tree.y_val)], verbose=False)
        proba = tree.model.predict_proba(tree.X_test)[:, 1]
        labels = np.asarray(tree.y_test).astype(int)
        out.append({
            "fold": int(fold["fold"]),
            "auroc": float(roc_auc_score(labels, proba)),
            "val_auroc": float(study.best_value),
            "n_features": int(tree.X_train.shape[1]),
            "n_test": int(len(labels)),
            "n_pos_test": int(labels.sum()),
            "seconds": round(time.time() - t0, 1),
            "best_params": json.dumps(study.best_params, default=float),
            "dates": [str(pd.Timestamp(d).date()) for d in fold["test"]["date"]],
            "labels": labels,
            "proba": proba,
        })
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def append_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def save_run(fold_results, keys: dict, baseline: dict, cfg, out_dir: Path,
             n_injected: int | None) -> pd.DataFrame:
    """Append one model's fold results and its test predictions."""
    rows, preds = [], []
    for res in fold_results:
        base = baseline.get(res["fold"], np.nan)
        rows.append({
            "experiment": cfg.experiment, "dataset": cfg.dataset_name, **keys,
            "fold": res["fold"], "auroc": res["auroc"], "auroc_baseline": base,
            "gain": res["auroc"] - base, "val_auroc": res["val_auroc"],
            "n_injected_total": n_injected, "n_features": res["n_features"],
            "n_test": res["n_test"], "n_pos_test": res["n_pos_test"],
            "num_trials": cfg.num_trials, "seconds": res["seconds"],
            "best_params": res["best_params"],
        })
        preds.append(pd.DataFrame({
            **{k: v for k, v in keys.items()}, "fold": res["fold"],
            "date": res["dates"], "label": res["labels"], "proba": res["proba"],
        }))
    rows = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    append_csv(out_dir / "results.csv", rows)
    append_csv(out_dir / "predictions.csv", pd.concat(preds, ignore_index=True))
    return rows


def completed_keys(results_csv: Path) -> set[tuple]:
    """(model, pool, background, lambda0, seed) groups with all three folds saved."""
    if not results_csv.exists():
        return set()
    done = pd.read_csv(results_csv)
    counts = done.groupby(KEY_COLUMNS, dropna=False)["fold"].nunique()
    return {tuple(k) for k, n in counts.items() if n == 3}


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def injection_pool(llm: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Tweets injected: informative ones (experiment) or non-informative ones (volume control).

    Both keep ``day``, the original local posting date, for reporting only: the
    injection draws every day from the whole pool.
    """
    if kind == "informative":
        return build_pool(llm)
    sel = (llm["status"] == "ok") & ~llm["informative"]
    pool = llm.loc[sel, ["tweet_id", "day", "lang"]].drop_duplicates("tweet_id")
    print(f"[pool] volume control: {len(pool)} non-informative tweets.")
    return pool.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)


def write_pool_report(out_dir: Path, pool: pd.DataFrame, pool_name: str) -> dict:
    """Print/save the single-pool description to ``outputs/<experiment>/pool_report.json``.

    The pool is unrestricted, so the report records its size and date span rather
    than any per-day eligibility; the JSON lets a finished run be audited afterwards.
    """
    report = {"pool": pool_name, **pool_report(pool)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / POOL_REPORT_FILE
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[pool_report] wrote {path}")
    return report


def run_baseline(cfg, out_dir: Path) -> dict[int, float]:
    """Train/evaluate the no-tweet baseline, or reuse it from results.csv."""
    results_csv = out_dir / "results.csv"
    if results_csv.exists():
        prev = pd.read_csv(results_csv)
        prev = prev[prev["model"] == "baseline"]
        if prev["fold"].nunique() == 3:
            aurocs = dict(zip(prev["fold"].astype(int), prev["auroc"]))
            print(f"[baseline] reused from {results_csv.name}: "
                  + " ".join(f"f{k}={v:.4f}" for k, v in aurocs.items()))
            return aurocs

    frame = pd.read_pickle(cfg.reference_pkl)
    frame["date"] = pd.to_datetime(frame["date"])
    res = tune_and_evaluate(frame, "baseline", cfg, out_dir / "_optuna" / "baseline")
    keys = {"model": "baseline", "pool": None, "background": None, "lambda0": np.nan, "seed": np.nan}
    save_run(res, keys, {}, cfg, out_dir, n_injected=None)
    aurocs = {r["fold"]: r["auroc"] for r in res}
    print("[baseline] test AUROC " + " ".join(f"f{k}={v:.4f}" for k, v in aurocs.items())
          + f" | mean={np.mean(list(aurocs.values())):.4f}")
    return aurocs


def previous_window(out_dir: Path) -> int | None:
    """Window recorded by an earlier run in this output directory.

    A ``config.json`` predating the ``--feature_window`` option means the run
    used the real pipeline's 7-day window, so that is what gets reported.
    """
    config = out_dir / "config.json"
    if not config.exists():
        return None
    cfg = json.loads(config.read_text(encoding="utf-8"))
    if "feature_window" in cfg:
        return int(cfg["feature_window"])
    if "feature_lags" in cfg:
        return len(cfg["feature_lags"]) + 1
    return 1 + len(NLP_FEATURE_LAGS)


#: config keys whose change invalidates an existing results.csv, with the value an
#: older config.json is assumed to have used when the key is missing entirely.
#: `injection_mode` is absent from every pre-spread run, which spread nothing,
#: i.e. used 'step'; such a run also passed no delta and so used SURGE_THRESHOLD.
#: Every pre-2026-11-02 run used the XGBoost estimator, so a config without the
#: key is assumed to have used it.
CONFIG_GUARD_KEYS = {"injection_mode": STEP_MODE, "delta": SURGE_THRESHOLD,
                     "estimator": "xgboost"}


def _prior_spread_horizon(prior: dict) -> int | None:
    """The spread offset an older config.json used, or None if it spread nothing.

    Window mode gained the ``horizon`` offset after the first window runs, which
    centred the spread on the surge day itself (offset 0). A config predating
    ``spread_horizon`` therefore used 0 when it ran window mode; step and ratio
    modes never used an offset, so nothing can drift for them.
    """
    if "spread_horizon" in prior:
        return prior["spread_horizon"]
    if prior.get("injection_mode", STEP_MODE) == WINDOW_MODE:
        return 0
    return None


def current_spread_horizon(cfg) -> int | None:
    """The spread offset this run uses, or None when the mode ignores it."""
    return HORIZON if cfg.injection_mode == WINDOW_MODE else None


def check_config_compatibility(out_dir: Path, cfg) -> None:
    """Abort if an existing results.csv was written under a different injection model.

    ``injection_mode``, ``delta`` and the window mode's spread offset change what
    every injected count means, and results.csv has no column for any of them, so
    resuming across a change would blend two incompatible injections in one
    table. Same reasoning as the :func:`previous_window` guard for
    ``--feature_window``.
    """
    config = out_dir / "config.json"
    if not (config.exists() and (out_dir / "results.csv").exists()):
        return
    prior = json.loads(config.read_text(encoding="utf-8"))
    checks = {key: (prior.get(key, legacy) if key in prior else legacy)
              for key, legacy in CONFIG_GUARD_KEYS.items()}
    checks["spread_horizon"] = _prior_spread_horizon(prior)
    now = {key: getattr(cfg, key) for key in CONFIG_GUARD_KEYS}
    now["spread_horizon"] = current_spread_horizon(cfg)
    for key, was in checks.items():
        if was is None or now[key] is None:
            continue
        if was != now[key]:
            raise SystemExit(
                f"outputs/{cfg.experiment}/results.csv was written with "
                f"--{key} {was!r}, this run asks for {now[key]!r}. Use a new "
                "--experiment name to keep the two runs comparable."
            )


def run_experiment(cfg) -> None:
    out_dir = OUTPUTS_DIR / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    days = reference_days(cfg.reference_pkl)

    # Number of days of tweet history every feature row covers (t, t-1, ...).
    # Passed explicitly so config.json records it and a resumed run cannot mix
    # two windows unnoticed.
    if cfg.feature_window < 1:
        raise ValueError(f"--feature_window must be >= 1, got {cfg.feature_window}.")
    lags = tuple(range(1, cfg.feature_window))

    # Resuming appends to results.csv, which has no window column: refuse to mix
    # two windows in one output directory.
    prior = previous_window(out_dir)
    if prior is not None and prior != cfg.feature_window and (out_dir / "results.csv").exists():
        raise SystemExit(
            f"outputs/{cfg.experiment}/results.csv was written with "
            f"--feature_window {prior}, this run asks for {cfg.feature_window}. "
            "Use a new --experiment name to keep the two runs comparable."
        )
    check_config_compatibility(out_dir, cfg)

    # BERT: one PCA fitted on the whole per-tweet embedding cache, before any
    # injection, so the reduced basis is identical for every fold, seed, lambda
    # and split. The lexicon path has no embeddings to reduce.
    cfg.bert_pca_reducer = None
    method_kwargs = {"lags": lags}
    if cfg.method == "bert":
        cfg.bert_pca_reducer = fit_bert_reducer(cfg.pca_threshold)
        method_kwargs["reducer"] = cfg.bert_pca_reducer

    (out_dir / "config.json").write_text(json.dumps(
        {**{k: v for k, v in vars(cfg).items() if k != "bert_pca_reducer"},
         "reference_pkl": str(cfg.reference_pkl), "horizon": HORIZON,
         "spread_horizon": current_spread_horizon(cfg),
         "feature_lags": list(lags),
         "model_seed": MODEL_SEED, "n_jobs": N_JOBS,
         "bert_pca": (cfg.bert_pca_reducer.describe()
                      if cfg.bert_pca_reducer is not None else
                      {"fit_population": "per_train_fold"})}, indent=2) + "\n")

    # 1. optional reference model (only needed for the gain columns)
    baseline = run_baseline(cfg, out_dir) if cfg.with_baseline else {}

    # shared inputs of every synthetic dataset
    llm = load_llm_labels()
    background = build_background(load_unique_tweets(), llm, days,
                                  keep_real_informative=cfg.keep_real_informative)
    pool = injection_pool(llm, cfg.pool)
    ratios = surge_ratio(days)
    bg_name = "keep_real_informative" if cfg.keep_real_informative else "default"

    # size and span of the one pool every day draws from
    write_pool_report(out_dir, pool, cfg.pool)
    if cfg.injection_mode == WINDOW_MODE:
        # The spread raises more days than the label marks, so the per-day
        # contrast is weaker than step mode at the same lambda0. Report it: a
        # null result at a given lambda is only interpretable against this.
        coverage = surge_window_coverage(ratios["label"].to_numpy(),
                                         cfg.feature_window, HORIZON)
        print(
            f"[run_experiment] spread over a {cfg.feature_window}-day window "
            f"offset by h={HORIZON}: "
            f"{(coverage > 0).mean():.3f} of days at rate lambda0*(1+{cfg.delta:g}) "
            f"vs label positive on {ratios['label'].mean():.3f} of days; "
            f"a day covers {coverage.mean():.2f} surges on average."
        )

    done = completed_keys(out_dir / "results.csv")
    for lambda0 in sorted(cfg.lambdas):
        for seed in cfg.seeds:
            keys = {"model": cfg.method, "pool": cfg.pool, "background": bg_name,
                    "lambda0": float(lambda0), "seed": int(seed)}
            if tuple(keys.values()) in done:
                print(f"[skip] lambda0={lambda0:g} seed={seed}: already in results.csv")
                continue

            # 2. synthetic dataset
            t0 = time.time()
            corpus, _, info = inject(background, pool, lambda0, seed, days=days,
                                     ratios=ratios, mode=cfg.injection_mode,
                                     delta=cfg.delta, window=cfg.feature_window)
            frame = build_daily_features(corpus, days, cfg.method,
                                       reference_pkl=cfg.reference_pkl, **method_kwargs)
            if cfg.save_datasets:
                ds_dir = out_dir / "datasets"
                ds_dir.mkdir(exist_ok=True)
                stem = Path(cfg.dataset_name).stem
                frame.to_pickle(ds_dir / f"{stem}__{cfg.method}_{cfg.pool}_lambda{lambda0:g}_seed{seed}.pkl")

            # 3. tune + evaluate
            res = tune_and_evaluate(frame, cfg.method, cfg,
                                    out_dir / "_optuna" / f"{cfg.method}_l{lambda0:g}_s{seed}")
            rows = save_run(res, keys, baseline, cfg, out_dir, info["n_injected_total"])

            # 4. test AUROC per fold (plus the gain, when a baseline was trained)
            tail = "" if not cfg.with_baseline else \
                " | mean gain=" + f"{rows['gain'].mean():+.4f}"
            print(f"[{cfg.method}|{cfg.pool}] lambda0={lambda0:g} seed={seed} "
                  f"n_injected={info['n_injected_total']} ({time.time() - t0:.0f} s) | "
                  + " ".join(f"f{r.fold}: {r.auroc:.4f}" for r in rows.itertuples())
                  + f" | mean AUROC={rows['auroc'].mean():.4f}" + tail, flush=True)

    # summary over seeds of the fold-averaged test AUROC
    res = pd.read_csv(out_dir / "results.csv")
    res = res[(res["model"] == cfg.method) & (res["pool"] == cfg.pool) & (res["background"] == bg_name)]
    if len(res):
        per_seed = res.groupby(["lambda0", "seed"])["auroc"].mean()
        summary = per_seed.groupby("lambda0").agg(["median", "min", "max", "count"])
        summary.columns = ["auroc_median", "auroc_min", "auroc_max", "n_seeds"]
        print("\n[summary] fold-averaged test AUROC (tweet features only), across seeds:")
        print(summary.round(4).to_string())
        if cfg.with_baseline:
            per_seed = res.groupby(["lambda0", "seed"])["gain"].mean()
            gains = per_seed.groupby("lambda0").agg(["median", "min", "max"])
            gains.columns = ["gain_median", "gain_min", "gain_max"]
            print("\n[summary] fold-averaged AUROC gain over the baseline, across seeds:")
            print(gains.round(4).to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Semi-synthetic tweet-injection experiment.")
    p.add_argument("--experiment", default='bert_xgboost', help="Name of the run; outputs go to outputs/<experiment>/.")
    p.add_argument("--dataset_name", default=REFERENCE_DATASET_PKL.name,
                   help="Reference dataset in FINAL_DATASET/ (dispatch features + labels); "
                        "also the prefix of saved synthetic datasets.")
    p.add_argument("--method", choices=["lexicon", "bert"], default="lexicon")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.1, 0.5, 1, 2.5, 5, 10, 20]), 
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pool", choices=["informative", "non_informative"], default="informative",
                   help="non_informative = volume control: same counts, uninformative tweets.")
    p.add_argument("--keep_real_informative", action="store_true")
    p.add_argument("--num_trials", type=int, default=5)
    p.add_argument("--pca_threshold", type=float, default=0.75, help="Used only by the BERT approach.")
    p.add_argument("--feature_window", type=int, default=FEATURE_WINDOW,
                   help="Days of tweet history per feature row: the reference day t plus "
                        "window-1 lags. Default 3 (t, t-1, t-2). "
                        "Also the number of days a surge spreads over in window mode.")
    p.add_argument("--injection_mode", choices=INJECTION_MODES, default=STEP_MODE,
                   help="window: elevated rate on the --feature_window days ending "
                        "horizon days before the surge (i.e. day d is raised if any "
                        "surge sits in d+h..d+h+w-1), capped at lambda0*(1+delta). "
                        "step: elevated rate on the surge day only. "
                        "ratio: rate = lambda0 * r_t.")
    p.add_argument("--delta", type=float, default=SURGE_THRESHOLD,
                   help="Surge multiplier of the injection rate, "
                        "used by the window and step modes.")

    p.add_argument("--with_baseline", action="store_true",
                   help="Also train the dispatch-history-only reference model and fill the "
                        "auroc_baseline/gain columns of results.csv. Off by default: the "
                        "tweet models are scored on their own absolute test AUROC.")
    p.add_argument("--estimator", choices=sorted(ESTIMATORS), default="xgboost",
                   help="Model class fitted on the features: 'xgboost' (default), "
                        "'logistic' (elastic-net logistic regression) or 'lightgbm' "
                        "(gradient-boosted trees, leaf-wise growth) — all on the same "
                        "features, folds, Optuna budget and metric. One estimator per "
                        "--experiment directory: results.csv has no estimator column, so "
                        "a resumed run with another estimator aborts.")
    p.add_argument("--save_datasets", action="store_true",
                   help="Pickle every synthetic dataset (about 25 MB each for BERT).")
    cfg = p.parse_args(argv)

    cfg.reference_pkl = DATASET_DIR / cfg.dataset_name
    if not cfg.reference_pkl.exists():
        p.error(f"{cfg.reference_pkl} does not exist.")
    if f"horizon_{HORIZON}" not in cfg.dataset_name:
        p.error(f"--dataset_name must match the injection horizon h={HORIZON} (paths.HORIZON).")
    return cfg


if __name__ == "__main__":
    run_experiment(parse_args()), 


