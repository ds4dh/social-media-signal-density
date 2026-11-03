"""Test AUROC against the injection density lambda0, one curve per classifier.

One figure per tweet representation (e.g. lexicon, BERT), each showing:
- the test AUROC of XGBoost, LightGBM and logistic regression ("Regression"),
  averaged over folds for each seed, then over seeds (band = seed range,
  faint dots = individual seeds);
- the theoretical bound of the paper for a window of w days (default w = 1);
- the chance level (dotted).

Results are read from  <outputs_dir>/<EXPERIMENT>_<MODEL>/results.csv,
with EXPERIMENT in {"bert", "lexicon"} and MODEL in {"xgboost", "lightgbm", "logistic"}.

Self-contained: no import from auroc_vs_lambda.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scipy.stats import norm, poisson

# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
ACCENTS = ["#2F5D8C", "#C46A3A", "#5B8E7D"]   # muted slate blue, terracotta, sage
INK, GREY, LIGHT, GRID = "#1f1f1e", "#8a8984", "#c9c8c2", "#ecebe7"

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": LIGHT, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "text.color": INK,
    "xtick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 0, "xtick.minor.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "regular", "axes.titlelocation": "left", "axes.titlepad": 6,
    "pdf.fonttype": 42, "svg.fonttype": "none",  # editable text in PDF/SVG
}

SERIES_KEYS = ["experiment", "model", "pool", "background"]
Y_LABELS = {"auroc": "AUROC", "gain": "AUROC gain"}


# ---------------------------------------------------------------------------
# theoretical bound
# ---------------------------------------------------------------------------
def optimal_auroc(lam, delta: float = 0.2, window: int = 1, method: str = "exact") -> np.ndarray:
    """Bound on the AUROC of any tweet score (Neyman-Pearson: the window count is optimal).

    Non-surge days: N0 ~ Poisson(lam * window); surge days: N1 ~ Poisson(lam * window * (1 + delta)).
    method="exact":  P(N1 > N0) + P(N1 = N0) / 2, summed over the Poisson pmfs.
    method="normal": Phi(delta * sqrt(lam * window / (2 + delta))), the normal approximation
                     of the paper (inaccurate when lam * window is small).
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    if method == "normal":
        return norm.cdf(delta * np.sqrt(lam * window / (2 + delta)))
    if method != "exact":
        raise ValueError("method must be 'exact' or 'normal'")
    out = np.empty_like(lam)
    for i, l in enumerate(lam):
        m0, m1 = l * window, l * window * (1 + delta)
        if m0 <= 0:
            out[i] = 0.5
            continue
        k = np.arange(int(poisson.ppf(1 - 1e-12, m1)) + 2)
        out[i] = np.sum(poisson.pmf(k, m1) * (poisson.cdf(k - 1, m0) + 0.5 * poisson.pmf(k, m0)))
    return out


def _dense_lambda_grid(grid: np.ndarray, n: int = 200) -> np.ndarray:
    """Smooth x values for the bound curve, covering exactly the plotted x range."""
    positive = grid[grid > 0]
    lo, hi = positive[0], positive[-1] * 1.4
    head = np.linspace(-0.25 * lo, lo, 20) if grid[0] == 0 else np.array([])
    xs = np.unique(np.concatenate([head, np.geomspace(lo, hi, n)]))
    return np.clip(xs, 0, None)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _spread(positions, min_gap):
    """Nudge label y-positions apart so none are closer than min_gap (keeps order)."""
    order = np.argsort(positions)
    placed = np.array(positions, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    result = np.empty_like(placed)
    result[order] = placed - (placed.mean() - np.mean(positions))
    return result


def _right_labels(ax, labels, min_gap_fraction=0.07):
    """labels: list of (text, y, colour). Drawn just right of the axes, de-overlapped."""
    labels = [lab for lab in labels if lab[0]]
    if not labels:
        return
    texts, ys, colours = zip(*labels)
    low, high = ax.get_ylim()
    for text, y, colour in zip(texts, _spread(ys, min_gap_fraction * (high - low)), colours):
        ax.annotate(text, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=colour,
                    annotation_clip=False)


def _jitter(n, seed=0, width=0.06):
    """Multiplicative jitter for a log axis."""
    return np.exp(np.random.default_rng(seed).uniform(-width, width, n))


def _style_lambda_axis(ax, grid):
    """Symmetric-log x axis: linear between 0 and the smallest positive lambda, log above,
    so lambda0 = 0 sits on the same continuous curve as the other values."""
    positive = grid[grid > 0]
    ax.set_xscale("symlog", linthresh=positive[0], linscale=0.5)
    ax.set_xticks(list(grid), [f"{v:g}" for v in grid])
    ax.minorticks_off()
    left = -0.25 * positive[0] if grid[0] == 0 else positive[0] / 1.4
    ax.set_xlim(left, positive[-1] * 1.4)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)


def _save(fig, path):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=800, bbox_inches="tight")
        print(f"Saved {path.with_suffix('.pdf')} and .png")


# directory suffix -> label shown in the figure
MODEL_LABELS = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "logistic": "Regression"}
# also accept the misspelt directory name
DIR_ALIASES = {"lightgbm": ["lightgbm", "lighgbm"]}
MODEL_ORDER = ["XGBoost", "LightGBM", "Regression"]
MODEL_COLOURS = dict(zip(MODEL_ORDER, ACCENTS))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_experiment(outputs_dir, experiment: str, models=tuple(MODEL_LABELS),
                    max_lambda: float | None = None) -> pd.DataFrame:
    """Concatenate the results.csv of every <experiment>_<model> run, labelled by model."""
    frames = []
    for model in models:
        candidates = [Path(outputs_dir) / f"{experiment}_{name}"
                      for name in DIR_ALIASES.get(model, [model])]
        run_dir = next((d for d in candidates if (d / "results.csv").is_file()), None)
        if run_dir is None:
            warnings.warn(f"No results.csv for {experiment}/{model} (looked in {candidates}).")
            continue
        df = pd.read_csv(run_dir / "results.csv")
        df = df[df["model"] != "baseline"].copy()
        df["label"] = MODEL_LABELS[model]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No results found for experiment '{experiment}' in {outputs_dir}.")
    results = pd.concat(frames, ignore_index=True)
    if max_lambda is not None:
        results = results[results["lambda0"] <= max_lambda]
    return results


def summarise_by_model(results: pd.DataFrame, metric: str = "auroc"):
    """Fold-averaged metric per (model, lambda0, seed), then mean / min / max over seeds."""
    folds = results.groupby(["label", "lambda0", "seed"])["fold"].nunique()
    incomplete = folds[folds != folds.max()]
    if len(incomplete):
        warnings.warn(f"{len(incomplete)} (model, lambda0, seed) group(s) have missing folds:\n"
                      + incomplete.to_string())
    per_seed = (results.groupby(["label", "lambda0", "seed"])[metric].mean()
                .rename("value").reset_index())
    summary = (per_seed.groupby(["label", "lambda0"])["value"]
               .agg(mean="mean", lo="min", hi="max", n_seeds="count").reset_index())
    return per_seed, summary


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def draw_models_vs_lambda(ax, results: pd.DataFrame, *, title: str | None = None,
                          bound: str | None = "exact", bound_delta: float = 0.2,
                          bound_window: int = 1, show_seeds: bool = True) -> list:
    """Draw one panel; returns the (text, y, colour) direct labels of its curves."""
    per_seed, summary = summarise_by_model(results)
    grid = np.sort(summary["lambda0"].unique())
    labels = []

    for name in [m for m in MODEL_ORDER if m in set(summary["label"])]:
        colour = MODEL_COLOURS[name]
        curve = summary[summary["label"] == name].sort_values("lambda0")
        if show_seeds:
            pts = per_seed[per_seed["label"] == name]
            ax.scatter(pts["lambda0"] * _jitter(len(pts)), pts["value"], s=4, color=colour,
                       alpha=0.22, lw=0, zorder=2)
        ax.fill_between(curve["lambda0"], curve["lo"], curve["hi"], color=colour,
                        alpha=0.14, lw=0, zorder=1)
        ax.plot(curve["lambda0"], curve["mean"], color=colour, lw=1.6, zorder=4,
                marker="o", ms=3, mfc=colour, mec="white", mew=0.6)
        labels.append((name, curve["mean"].iloc[-1], colour))

    ax.axhline(0.5, color=GREY, ls=(0, (1, 2)), lw=0.8, zorder=1)
    if bound:
        xs = _dense_lambda_grid(grid)
        ys = optimal_auroc(xs, bound_delta, bound_window, bound)
        ax.plot(xs, ys, color=INK, lw=0.9, zorder=3)
        labels.append(("Theoretical bound", ys[-1], INK))

    _style_lambda_axis(ax, grid)
    ax.set_xlabel(r"$\lambda_0$")
    ax.set_ylabel("AUROC")
    if title:
        ax.set_title(title, loc="center", fontsize=9, fontweight="bold")
    return labels


def _auto_ylim(axes, low: float = 0.45, pad: float = 0.03, step: float = 0.05) -> tuple[float, float]:
    """Common y range: from `low` to just above the highest curve, rounded up to `step`."""
    top = max(ax.dataLim.ymax for ax in np.atleast_1d(axes))
    return low, float(np.ceil((top + pad) / step) * step)


def plot_models_vs_lambda(results: pd.DataFrame, path=None, *, title: str | None = None,
                          bound: str | None = "exact", bound_delta: float = 0.2,
                          bound_window: int = 1, show_seeds: bool = True,
                          ylim: tuple[float, float] | None = None,
                          figsize=(4.4, 2.3)) -> plt.Figure:
    """One figure for one representation: three classifiers + bound, direct labels."""
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axisbelow(True)
        labels = draw_models_vs_lambda(ax, results, title=title, bound=bound,
                                       bound_delta=bound_delta, bound_window=bound_window,
                                       show_seeds=show_seeds)
        ax.set_ylim(*(ylim or _auto_ylim(ax)))
        _right_labels(ax, labels)
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_side_by_side(results_by_title: dict[str, pd.DataFrame], path=None, *,
                      bound: str | None = "exact", bound_delta: float = 0.2,
                      bound_window: int = 1, show_seeds: bool = True,
                      ylim: tuple[float, float] | None = None,
                      figsize=(7.2, 2.3)) -> plt.Figure:
    """Optional paper figure: one panel per representation, shared y axis and legend."""
    with plt.rc_context(PAPER_STYLE):
        fig, axes = plt.subplots(1, len(results_by_title), figsize=figsize, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (title, results) in zip(axes, results_by_title.items()):
            ax.set_axisbelow(True)
            draw_models_vs_lambda(ax, results, title=title, bound=bound,
                                  bound_delta=bound_delta, bound_window=bound_window,
                                  show_seeds=show_seeds)
        axes[0].set_ylim(*(ylim or _auto_ylim(axes)))   # shared y axis
        for ax in axes[1:]:
            ax.set_ylabel("")
        handles = [Line2D([], [], color=MODEL_COLOURS[m], lw=1.6, marker="o", ms=3,
                          mec="white", mew=0.6, label=m) for m in MODEL_ORDER]
        if bound:
            handles.append(Line2D([], [], color=INK, lw=0.9, label="Theoretical bound"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        _save(fig, path)
    return fig

# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def auroc_table(results_by_title: dict[str, pd.DataFrame], *, bound: str | None = "exact",
                bound_delta: float = 0.2, bound_window: int = 1, show_range: bool = True,
                decimals: int = 3, print_table: bool = True) -> pd.DataFrame:
    """Mean test AUROC (folds, then seeds) per representation, model and lambda0.

    Returns a long DataFrame (representation, model, lambda0, mean, lo, hi, n_seeds) and,
    if print_table, prints one table per representation: rows = lambda0, columns = models
    (+ the theoretical bound), cells = mean [seed min-max].
    """
    rows = []
    for title, results in results_by_title.items():
        _, summary = summarise_by_model(results)
        rows.append(summary.rename(columns={"label": "model"}).assign(representation=title))
    long = pd.concat(rows, ignore_index=True)[
        ["representation", "model", "lambda0", "mean", "lo", "hi", "n_seeds"]]

    if print_table:
        fmt = f"{{:.{decimals}f}}"
        for title, sub in long.groupby("representation", sort=False):
            cells = sub.assign(cell=[
                fmt.format(m) + (f" [{fmt.format(lo)}-{fmt.format(hi)}]" if show_range else "")
                for m, lo, hi in zip(sub["mean"], sub["lo"], sub["hi"])])
            table = cells.pivot(index="lambda0", columns="model", values="cell")
            table = table[[m for m in MODEL_ORDER if m in table.columns]]
            if bound:
                table["Bound"] = [fmt.format(v) for v in
                                  optimal_auroc(table.index.to_numpy(float), bound_delta,
                                                bound_window, bound)]
            print(f"\n=== {title}: mean test AUROC (folds, then seeds)"
                  + (" [seed min-max]" if show_range else "") + " ===")
            print(table.to_string())
    return long