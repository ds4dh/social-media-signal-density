"""Theoretical ceiling on the AUROC gain from tweets, as a function of λ.

Model (Task.md section 7). Over a window of ``w`` days the number of informative
tweets is Poisson: ``N0 ~ Poisson(λ·w)`` on a day drawn from the ordinary
distribution, and ``N1 ~ Poisson((1+δ)·λ·w)`` on a day with elevated emergency
demand, δ being the relative excess. The best a tweet-only score can do is the
AUROC of the likelihood-ratio test between those two counts, which for integer
counts resolves to the tie-aware expression

    auroc_max_exact(λ) = Σ_k P1(k) · [ P0(N < k) + ½·P0(k) ]           (1)

i.e. a tie between equal counts is worth a coin flip. Equation (1) is compared
against the closed-form normal approximation

    auroc_max_approx(λ) = Φ( δ·sqrt(λ·w / (2+δ)) )                      (2)

obtained from ``N1 - N0``, whose mean is ``δ·λ·w`` and whose variance is the sum
``(2+δ)·λ·w`` because the two counts are independent.

Assuming the call-history score and the tweet score are independent signals with
probits ``z_c`` and ``z_t``, adding tweets to the calls raises the AUROC to
``Φ(sqrt(z_c² + z_t²))``, so the achievable gain is that value minus the
baseline AUROC (3). ``lambda_for_gain`` inverts (3) by bisection to report how
dense the informative tweets would have to be for a given gain.

λ here is a daily rate, so λ, ``--lambda`` and every λ in the outputs are
informative tweets per day.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, poisson

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = PACKAGE_DIR / "outputs"
SUMMARY_FILE = "bound_summary.json"
PARAMS_FILE = "bound_params.json"
FIGURE_FILE = "fig_bound.png"
LAMBDA_SUMMARY_FILE = "lambda_summary.json"

DELTA_DEFAULT = 0.2
WINDOW_DEFAULT = 7
BASELINE_AUROC_DEFAULT = 0.6757
TAIL_TOL = 1e-12
GAINS = (0.01, 0.03, 0.05)
LAMBDA_GRID = np.logspace(-2, 2, 121)
BOUNDS = (1e-4, 1e4)
ONE_MINUS_EPS = 1.0 - 1e-12


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--delta", type=float, default=DELTA_DEFAULT, help="Relative excess of informative tweets on high-demand days.")
    parser.add_argument("--window", type=int, default=WINDOW_DEFAULT, help="Window length in days.")
    parser.add_argument("--baseline_auroc", type=float, default=BASELINE_AUROC_DEFAULT, help="AUROC of the call-history baseline.")
    parser.add_argument(
        "--lambda", dest="lambda_", type=float, default=None,
        help="Observed λ (informative tweets/day). Default: lambda_up for Geneva in lambda_summary.json.",
    )
    parser.add_argument("--lambda_summary", type=Path, default=OUTPUT_DIR / LAMBDA_SUMMARY_FILE)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no_figure", action="store_true", help="Skip fig_bound.png.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# (1) exact ceiling
# ---------------------------------------------------------------------------


def auroc_max_exact(
    lam: float,
    delta: float = DELTA_DEFAULT,
    window: int = WINDOW_DEFAULT,
    tol: float = TAIL_TOL,
) -> float:
    """Exact tweet-only AUROC ceiling from equation (1)."""
    if lam <= 0:
        return 0.5
    mu0 = lam * window
    mu1 = (1.0 + delta) * lam * window
    kmax = _truncate_index(mu1, tol)
    k = np.arange(kmax + 1)
    p1 = poisson.pmf(k, mu1)
    p0 = poisson.pmf(k, mu0)
    below = poisson.cdf(k - 1, mu0)
    return float(np.dot(p1, below + 0.5 * p0))


def _truncate_index(mu: float, tol: float) -> int:
    """Smallest k with ``P(X > k) < tol`` for ``X ~ Poisson(mu)``, plus a margin."""
    kmax = int(max(poisson.isf(tol, mu), mu)) + 10
    while poisson.sf(kmax, mu) >= tol:
        kmax *= 2
    return kmax


# ---------------------------------------------------------------------------
# (2) normal approximation, (3) gain, and the inversion
# ---------------------------------------------------------------------------


def auroc_max_approx(
    lam: float, delta: float = DELTA_DEFAULT, window: int = WINDOW_DEFAULT
) -> float:
    """Normal approximation (2) of the tweet-only ceiling."""
    if lam <= 0:
        return 0.5
    return float(norm.cdf(delta * math.sqrt(lam * window / (2.0 + delta))))


def combined_auroc(
    lam: float,
    baseline_auroc: float = BASELINE_AUROC_DEFAULT,
    exact: bool = True,
    **kwargs,
) -> float:
    """Maximal AUROC of calls + tweets, equation (3) without the baseline offset."""
    ceiling = auroc_max_exact(lam, **kwargs) if exact else auroc_max_approx(lam, **kwargs)
    # probit of a probability that is exactly 0.5 or 1.0 is 0 or unbounded.
    z_t = norm.ppf(min(max(ceiling, 0.5), ONE_MINUS_EPS))
    z_c = norm.ppf(min(max(baseline_auroc, 1e-12), ONE_MINUS_EPS))
    return float(norm.cdf(math.sqrt(z_c * z_c + z_t * z_t)))


def max_gain(
    lam: float,
    baseline_auroc: float = BASELINE_AUROC_DEFAULT,
    exact: bool = True,
    **kwargs,
) -> float:
    """AUROC gain available from tweets on top of the baseline, equation (3)."""
    return combined_auroc(lam, baseline_auroc, exact=exact, **kwargs) - baseline_auroc


def lambda_for_gain(
    gain: float,
    baseline_auroc: float = BASELINE_AUROC_DEFAULT,
    bounds: tuple[float, float] = BOUNDS,
    **kwargs,
) -> float | None:
    """Smallest λ on ``bounds`` whose gain reaches ``gain``; ``None`` if unreachable."""
    low, high = bounds
    g_low = max_gain(low, baseline_auroc, **kwargs)
    g_high = max_gain(high, baseline_auroc, **kwargs)
    if g_high < gain:
        return None
    if g_low >= gain:
        return low
    return float(brentq(lambda x: max_gain(x, baseline_auroc, **kwargs) - gain, low, high, xtol=1e-10, rtol=1e-10))


# ---------------------------------------------------------------------------
# observed lambda_up
# ---------------------------------------------------------------------------
def read_lambda_up(path: Path) -> dict[str, float]:
    """Pull the Geneva and corpus ``lambda_up`` out of section 6's summary.

    Accepts both ``{"geneva": {"lambda_up": x}, "corpus": {...}}`` and the flat
    ``{"lambda_up_geneva": x, ...}`` spelling.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run estimate_lambda.py first, or pass --lambda explicitly."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for scope, aliases in (
        ("geneva", ("lambda_up_geneva", "geneva_lambda_up")),
        ("corpus", ("lambda_up_corpus", "corpus_lambda_up")),
    ):
        value = None
        nested = payload.get(scope)
        if isinstance(nested, dict):
            value = nested.get("lambda_up", nested.get("lambda_up_hard"))
        if value is None:
            for alias in aliases:
                if alias in payload:
                    value = payload[alias]
                    break
        if value is not None:
            out[scope] = float(value)
    if not out:
        raise KeyError(
            f"No lambda_up found in {path}. Expected a 'geneva'/'corpus' object "
            "with a 'lambda_up' key, or flat 'lambda_up_geneva'/'lambda_up_corpus' keys."
        )
    return out


def check_approximation(
    delta: float, window: int, baseline_auroc: float, lambda_up: dict[str, float]
) -> list[dict]:
    """Rows comparing (1), (2) and (3) over a log grid plus the observed λ values."""
    grid = [
        {"lambda": float(lam), "exact": auroc_max_exact(lam, delta, window),
         "approx": auroc_max_approx(lam, delta, window),
         "gain": max_gain(lam, baseline_auroc, delta=delta, window=window)}
        for lam in LAMBDA_GRID
    ]
    for scope, lam in lambda_up.items():
        grid.append(
            {"lambda": float(lam), "exact": auroc_max_exact(lam, delta, window),
             "approx": auroc_max_approx(lam, delta, window),
             "gain": max_gain(lam, baseline_auroc, delta=delta, window=window),
             "scope": scope}
        )
    return grid


def make_figure(rows: list[dict], baseline_auroc: float, lambda_up: dict[str, float], path: Path) -> None:
    """Plot the maximal AUROC against λ on a log axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = [r for r in rows if "scope" not in r]
    xs = [r["lambda"] for r in curve]
    ys = [baseline_auroc + r["gain"] for r in curve]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.semilogx(xs, ys, color="tab:blue", lw=1.8, label="calls + tweets (exact ceiling)")
    ax.axhline(baseline_auroc, color="grey", ls="--", lw=1.2, label=f"baseline AUROC = {baseline_auroc:.4f}")
    for (scope, lam), colour in zip(sorted(lambda_up.items()), ("tab:red", "tab:green")):
        row = next(r for r in rows if r.get("scope") == scope)
        ax.axvline(lam, color=colour, ls=":", lw=1.4, label=f"λ_up ({scope}) = {lam:.3g}")
        ax.plot([lam], [baseline_auroc + row["gain"]], "o", color=colour)
    ax.set_xlabel("λ informative tweets per day")
    ax.set_ylabel("maximal AUROC")
    ax.set_ylim(bottom=min(baseline_auroc - 0.005, min(ys)))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 0.5 < args.baseline_auroc < 1:
        raise SystemExit(f"--baseline_auroc must lie in (0.5, 1), got {args.baseline_auroc}.")
    if args.window <= 0:
        raise SystemExit(f"--window must be a positive number of days, got {args.window}.")
    if args.delta <= -1:
        raise SystemExit(f"--delta must exceed -1, got {args.delta}.")

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda_up: dict[str, float] = {}
    if args.lambda_ is not None:
        lambda_up["geneva"] = float(args.lambda_)
        print(f"Using --lambda {args.lambda_} as the Geneva density.")
    else:
        try:
            lambda_up = read_lambda_up(Path(args.lambda_summary))
        except (FileNotFoundError, KeyError) as exc:
            raise SystemExit(str(exc)) from exc
        print(f"λ_up read from {args.lambda_summary}: {lambda_up}")

    common = {"delta": args.delta, "window": args.window}
    rows = check_approximation(args.delta, args.window, args.baseline_auroc, lambda_up)

    print(f"\ndelta={args.delta} window={args.window} baseline_auroc={args.baseline_auroc}")
    print(f"{'lambda':>10} {'exact':>9} {'approx':>9} {'diff':>9} {'gain':>9}")
    for row in rows:
        marker = f"  <- {row['scope']}" if "scope" in row else ""
        print(
            f"{row['lambda']:10.4g} {row['exact']:9.5f} {row['approx']:9.5f} "
            f"{row['approx'] - row['exact']:9.5f} {row['gain']:9.5f}{marker}"
        )

    required = {}
    for g in GAINS:
        lam = lambda_for_gain(g, args.baseline_auroc, **common)
        required[f"{g:.2f}"] = lam
        print(f"lambda_min for gain {g:.2f}: {'> %g (not reached)' % BOUNDS[1] if lam is None else f'{lam:.4g}'}")

    observed = {}
    for scope, lam in lambda_up.items():
        observed[scope] = {
            "lambda": lam,
            "auroc_max_exact": auroc_max_exact(lam, **common),
            "auroc_max_approx": auroc_max_approx(lam, **common),
            "max_gain": max_gain(lam, args.baseline_auroc, **common),
            "maximal_auroc": combined_auroc(lam, args.baseline_auroc, **common),
        }

    summary = {
        "parameters": {"delta": args.delta, "window": args.window, "baseline_auroc": args.baseline_auroc,
                       "tail_tol": TAIL_TOL, "lambda_bounds": list(BOUNDS)},
        "lambda_up": lambda_up,
        "observed": observed,
        "lambda_min_for_gain": required,
        "max_abs_exact_minus_approx_lam_w_ge_1": max(
            (abs(r["approx"] - r["exact"]) for r in rows if r["lambda"] * args.window >= 1), default=None
        ),
    }
    (args.output_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / PARAMS_FILE).write_text(
        json.dumps({"script": "bound.py", "seed": 42, **summary["parameters"]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nSummary -> {args.output_dir / SUMMARY_FILE}")

    if not args.no_figure:
        figure_path = args.output_dir / FIGURE_FILE
        make_figure(rows, args.baseline_auroc, lambda_up, figure_path)
        print(f"Figure  -> {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
