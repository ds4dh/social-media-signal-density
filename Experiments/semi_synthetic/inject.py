"""Injection of resampled informative tweets (Task.md step 3).

This is the **only** module of the experiment that reads the future target
series, and it uses it exclusively to decide how many tweets land on each day.
The target never becomes a feature and never reaches a dataset.

For a density ``lambda0`` and a seed:

1. ``r_t = target[t+h] / mean(target[t-27 : t])`` — the ratio whose threshold
   defines the ``surge_detection`` label
   (``label_t = 1 iff r_t > 1 + surge_threshold``). Days where the label is
   undefined get ``r_t = 0``, i.e. no injection.
2. ``k_t ~ Poisson(rate_t)``, drawn independently per day. ``mode`` picks the
   rate; the default ``"window"`` is the one :mod:`run_experiment` uses:

   ``"window"`` (option A, capped): day ``d`` is elevated iff at least one surge
   day sits ``horizon`` rows ahead of it, in the ``window`` rows ending at
   ``d+h+w-1``::

       rate_d = lambda0 * (1 + delta)   if any y_t == 1 for t in [d+h, d+h+w-1]
              = lambda0                 otherwise

   Equivalently, a surge labelled on day ``t`` puts its elevated block on days
   ``t-h, t-h-1, ..., t-h-w+1``: the block is ``w`` days long and ends ``h``
   days *before* the surge day, so the injected tweets lead the label by the
   horizon instead of covering the surge day itself.

   The rate is **capped**, never stacked: a day covered by three overlapping
   surge windows gets the same rate as one covered by a single window, so the
   injection never adds signal beyond ``lambda0 * (1 + delta)``.
   ``"step"``: ``rate_t = lambda0 * (1 + delta * y_t)`` — the elevated count
   stays on the surge day itself.
   ``"ratio"``: ``rate_t = lambda0 * r_t`` — rate proportional to the surge
   ratio, so it scales with demand rather than with the binary label.

3. ``k_t`` tweets drawn **with replacement**, uniformly from the single pool of
   informative tweets spanning the whole period, assigned to day ``t``.

Step 3 draws from one pool shared by every day: the injection knows nothing
about the CV folds, and the posting date of a pool tweet plays no role in the day
it lands on. The consequence is accepted by design — a tweet drawn into a test
day of a fold can also sit on a training day of that same fold, so a measured
gain blends the signal the experiment wants to detect with the memorisation of a
few exact embeddings. :func:`pool_report` records what the days draw from: one
pool of every informative tweet, shared by all 1,071 reference days.

How much signal the day carries depends on the mode. In ``"window"`` and
``"step"`` a day whose target h days ahead is twice its baseline still injects
only ``1 + delta`` times ``lambda0`` tweets; only ``"ratio"`` tracks the size of
the surge.

Dilution is the cost of the ``"window"`` spread. A surge day pushes tweets
backwards into the days that precede it by h, and because surge days are adjacent
in time, those extra days are largely non-surge days: over the 1,071 reference
days a 3-day window offset by h=3 elevates 52.3 % of days to reach a label
positive on 29.9 % of them — and 88 surge days get no elevated rate at all,
because their own forward window [d+h, d+h+w-1] holds no surge. The mean rate
contrast between a surge day and a non-surge day is therefore *lower* in window
mode (1.05) than in step mode (1.20) at delta=0.2. The contrast is scale-free,
so this holds at every ``lambda0``.

Run: ``python inject.py --lambda0 1 10 100`` writes the injection tables.
"""

from __future__ import annotations

import argparse
import types

import numpy as np
import pandas as pd

from corpus import reference_days
from paths import (
    DISPATCH_UNIT,
    FEATURE_WINDOW,
    HORIZON,
    OUTPUTS_DIR,
    SURGE_THRESHOLD,
    TARGET_SCOPE,
)
from target_construction import (
    SURGE_BASELINE_WINDOW,
    SURGE_MIN_PERIODS,
    compute_labels,
)

#: column holding the injected count per day in the injection table
INJECTED_COL = "n_injected"

#: name of the pool report written next to the experiment results
POOL_REPORT_FILE = "pool_report.json"

#: injection modes accepted by :func:`injection_counts`, in the order of the
#: module docstring. ``WINDOW_MODE`` is the experiment default.
WINDOW_MODE, STEP_MODE, RATIO_MODE = "window", "step", "ratio"
INJECTION_MODES = (WINDOW_MODE, STEP_MODE, RATIO_MODE)


def surge_window_coverage(
    label: np.ndarray, window: int, horizon: int = HORIZON
) -> np.ndarray:
    """Number of surge days whose spread-window covers each day, per day.

    A surge day ``t`` is credited to the ``window`` days ending ``horizon`` days
    before it, so day ``d`` is covered by every surge day in
    ``d+horizon, ..., d+horizon+window-1``::

        surge at t  ->  elevates days t-horizon, ..., t-horizon-window+1

    ``horizon = HORIZON`` puts the whole spread one horizon **ahead** of the
    surge day, on exactly the rows that the surge's own label day can see: day
    ``d`` is labelled from ``target[d+h]``, so a surge landing on ``d+h`` is
    credited to rows ``d, ..., d-window+1`` — the rows day ``d``'s features
    read. The injected tweets are then a leading indicator of the surge rather
    than a copy of the surge day itself.

    With ``horizon = 0`` this degenerates to the surge day's own feed-window,
    ``t, ..., t-window+1``.

    Both the coverage count and the window are **row-based** over the
    reference-day table, matching ``features.add_row_lags``, which shifts rows
    rather than calendar dates. Days whose forward window runs past the last
    reference day see only the rows that exist; the padded tail is never
    silently dropped, so the output always has one value per input day.

    Only ``> 0`` matters for :func:`injection_rates` in window mode — the count
    is returned rather than a boolean so a caller can audit the overlap.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}.")
    y = np.asarray(label, dtype="float64")
    # Right-pad by `horizon` zeros: day d sums y over [d+horizon,
    # d+horizon+window-1], and rows past the end of the table do not exist and
    # contribute nothing. Without the pad the "full" convolution is too short
    # and the last `horizon` days would fall off the slice.
    padded = np.concatenate([y, np.zeros(int(horizon))])
    start = int(horizon) + int(window) - 1
    return np.convolve(padded, np.ones(int(window)), mode="full")[
        start:start + len(y)
    ]


def pool_report(pool: pd.DataFrame, *, verbose: bool = True) -> dict:
    """Describe the single pool that every reference day draws from.

    There is no restriction to report: the whole period's informative tweets are
    available to all 1,071 days, so the same tweet can land on a fold's training
    day and on that fold's test day. The record exists so a finished run shows
    what was shared, not to guard against the overlap.
    """
    days = pd.to_datetime(pool["day"])
    report = {
        "n_unique": int(pd.unique(pool["tweet_id"]).size),
        "first_day": str(days.min().date()),
        "last_day": str(days.max().date()),
        "draw_mode": "uniform_with_replacement_over_the_whole_pool",
    }
    if verbose:
        print(
            f"[pool_report] {report['n_unique']} unique relevant tweets shared by "
            f"every day ({report['first_day']}..{report['last_day']}); draws are "
            f"uniform and with replacement, across fold boundaries."
        )
    return report


def surge_ratio(
    days: list[str],
    *,
    horizon: int = HORIZON,
    surge_threshold: float = SURGE_THRESHOLD,
    unit: str = DISPATCH_UNIT,
    scope: str = TARGET_SCOPE,
    target_series: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ``date, r, baseline, future, label`` for ``days``.

    ``label`` comes from the pipeline's own ``compute_labels`` with strategy
    ``'surge_detection'``, so it is by construction the target variable the
    models are trained on. ``r`` is the ratio that threshold defines the label::

        r_t = target[t+h] / mean(target[t-27 : t])

    Days whose label is undefined (incomplete baseline window, future day
    outside the series) get ``r = 0`` and ``label = 0``, i.e. no injection;
    ``label`` is never NaN.
    """
    if target_series is None:
        from dispatch_data import daily_target_series

        params = types.SimpleNamespace(
            general_dispatch_center_focus=unit, target_scope=scope
        )
        t = daily_target_series(params)
    else:
        # DataFrame(df) copies: the caller's series must survive untouched (its
        # `date` column is datetime for the pipeline, and tests reuse it here).
        t = pd.DataFrame(target_series)
    t["date"] = pd.to_datetime(t["date"]).dt.strftime("%Y-%m-%d")
    t = t.sort_values("date").reset_index(drop=True)

    s = pd.Series(t["n_target"].to_numpy(dtype="float64"))
    # Exactly the quantities that threshold the label:
    # label_t = 1 iff future_t > (1 + surge_threshold) * baseline_t.
    baseline = s.rolling(SURGE_BASELINE_WINDOW, min_periods=SURGE_MIN_PERIODS).mean()
    future = s.shift(-horizon)
    label = pd.Series(
        compute_labels(s, pd.to_datetime(t["date"]), "surge_detection",
                       horizon=horizon, surge_threshold=surge_threshold),
        index=t.index,
    )

    ratio = pd.DataFrame({
        "date": t["date"].to_numpy(),
        "baseline": baseline.to_numpy(),
        "future": future.to_numpy(),
        "label": label.to_numpy(),
    })
    ratio["r"] = ratio["future"] / ratio["baseline"]
    undefined = label.isna().to_numpy() | ~np.isfinite(ratio["r"].to_numpy())
    ratio.loc[undefined, "r"] = 0.0
    ratio["r"] = ratio["r"].clip(lower=0.0)
    ratio.loc[undefined, "label"] = 0.0

    out = ratio[ratio["date"].isin(days)].sort_values("date").reset_index(drop=True)
    missing = set(days) - set(out["date"])
    if missing:
        raise ValueError(
            f"No target series for {len(missing)} day(s): {sorted(missing)[:5]}"
        )
    return out


def injection_rates(
    ratios: pd.DataFrame,
    lambda0: float,
    *,
    mode: str = WINDOW_MODE,
    delta: float = SURGE_THRESHOLD,
    window: int = FEATURE_WINDOW,
    horizon: int = HORIZON,
) -> pd.Series:
    """Mean injected tweets per day: the rate vector :func:`injection_counts` samples.

    ``mode="window"`` : a surge day spreads its elevated rate over the ``window``
                       rows that end ``horizon`` days before it, capped at
                       ``lambda0 * (1 + delta)``::

                           rate_d = lambda0 * (1 + delta) if any y_t == 1,
                                    t in [d + horizon, d + horizon + window - 1]
                                  = lambda0 otherwise

                       The elevated block therefore sits ``horizon`` days *ahead*
                       of the surge it belongs to, on the rows that the surge's
                       label day reads: day ``d`` is labelled from ``target[d+h]``,
                       so tweets on ``d, ..., d-window+1`` lead the surge at
                       ``d+h`` rather than repeating its own day.
                       Overlapping windows never stack, so the rate never exceeds
                       ``lambda0 * (1 + delta)`` and the injection never adds
                       signal beyond a single surge.
    ``mode="step"``   : ``lambda0 * (1 + delta * y_t)``, the model of the theory
                       section (``lambda0`` = non-surge rate), with the elevated
                       rate confined to the surge day itself.
    ``mode="ratio"``  : ``lambda0 * r_t``, with the surge ratio ``r_t = c_{t+h} /
                       mean(c_{t-27..t})``.

    Days whose label is undefined have ``label = 0`` (baseline ``lambda0`` in
    window and step modes) and ``r = 0`` (no injection in ratio mode).

    Requires ``ratios`` sorted by date: the window mode spreads over rows, so an
    unsorted frame would smear the signal onto unrelated days.
    """
    if lambda0 < 0:
        raise ValueError(f"lambda0 must be >= 0, got {lambda0}.")
    if mode in (WINDOW_MODE, STEP_MODE) and delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}.")
    label = np.asarray(ratios["label"], dtype="float64")
    if mode == WINDOW_MODE:
        if not pd.Index(ratios["date"]).is_monotonic_increasing:
            raise ValueError(
                "injection_rates with mode='window' needs `ratios` sorted by date: "
                "the spread shifts rows. Sort the frame or pass days=sorted(...)."
            )
        elevated = surge_window_coverage(label, window, horizon) > 0
        rates = float(lambda0) * (1.0 + float(delta) * elevated)
    elif mode == STEP_MODE:
        rates = float(lambda0) * (1.0 + float(delta) * label)
    elif mode == RATIO_MODE:
        rates = float(lambda0) * np.asarray(ratios["r"], dtype="float64")
    else:
        raise ValueError(
            f"Unknown injection mode {mode!r}; expected one of {INJECTION_MODES}."
        )
    return pd.Series(rates, index=pd.Index(ratios["date"]), name="rate")


def injection_counts(
    ratios: pd.DataFrame,
    lambda0: float,
    seed: int | np.random.SeedSequence | np.random.Generator,
    *,
    mode: str = WINDOW_MODE,
    delta: float = SURGE_THRESHOLD,
    window: int = FEATURE_WINDOW,
    horizon: int = HORIZON,
) -> pd.Series:
    """Draw ``k_t ~ Poisson(rate_t)``, one injected-tweet count per day.

    ``rate_t`` comes from :func:`injection_rates`; see it for the modes.

    ``seed`` may be an explicit :class:`numpy.random.Generator` — the pair from
    :func:`injection_streams` — so the counts and the pool draws use independent
    streams. One generator consumes the days in the order of ``ratios``, so the
    same ``(ratios, lambda0, seed, mode, delta, window, horizon)`` always yields
    the same counts.
    """
    rates = injection_rates(
        ratios, lambda0, mode=mode, delta=delta, window=window, horizon=horizon
    )
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.poisson(rates.to_numpy()), index=rates.index, name=INJECTED_COL
    )


def expected_injections(
    ratios: pd.DataFrame, lambda0: float, *, mode: str = WINDOW_MODE,
    delta: float = SURGE_THRESHOLD, window: int = FEATURE_WINDOW,
    horizon: int = HORIZON,
) -> float:
    """``sum_t rate_t`` — the expectation :func:`injection_counts` draws from.

    Used for the ``expected_total`` diagnostic, so a reported count is compared
    against the rate vector that was actually sampled rather than against a
    fixed-ratio guess that only holds in ratio mode.
    """
    return float(
        injection_rates(
            ratios, lambda0, mode=mode, delta=delta, window=window, horizon=horizon
        ).sum()
    )


def inject_pool_draws(pool, counts: pd.Series,
                      seed: int | np.random.SeedSequence) -> pd.DataFrame:
    """Draw ``counts[t]`` pool tweets with replacement for each day ``t``.

    ``pool`` is a DataFrame of ``tweet_id, day`` (the relevant tweets and the local
    date they were originally posted on) or a bare array of tweet ids. Every day
    draws uniformly from the **whole** pool: the posting date plays no role in the
    day a tweet lands on, and no day is restricted to a date window. A tweet drawn
    into a fold's test day can therefore also sit on a training day of that fold.

    Returns ``tweet_id, day`` — one row per injected tweet. The same tweet can be
    drawn repeatedly (sampling with replacement); the per-tweet features come from
    the cache by ``tweet_id``.

    Raises :class:`ValueError` only when the pool itself is empty.
    """
    if isinstance(pool, pd.DataFrame):
        if "tweet_id" not in pool.columns:
            raise ValueError("The pool needs a 'tweet_id' column.")
        pool_ids = pool["tweet_id"].to_numpy(dtype=object)
    else:
        pool_ids = np.asarray(pool, dtype=object)
    if not len(pool_ids):
        raise ValueError("The pool of informative tweets is empty; nothing to inject.")

    rng = np.random.default_rng(seed)
    ids, days = [], []
    for day, k in counts.items():
        k = int(k)
        if k <= 0:
            continue
        ids.append(pool_ids[rng.integers(0, len(pool_ids), size=k)])
        days.append(np.full(k, str(day), dtype=object))
    if not ids:
        return pd.DataFrame({"tweet_id": pd.Series(dtype=object), "day": pd.Series(dtype=object)})
    return pd.DataFrame({"tweet_id": np.concatenate(ids), "day": np.concatenate(days)})


def injection_streams(seed: int | np.random.SeedSequence) -> tuple[np.random.Generator, np.random.Generator]:
    """Return ``(count_stream, draw_stream)`` for an injection seed.

    The two are spawned children of ``SeedSequence(seed)``, so the pool draws no
    longer replay the Poisson-count stream from the same seed. Spawning both
    streams (rather than reusing the parent for the counts) is the canonical numpy
    pattern and is independent of call order.

    Consequence worth knowing: the counts are therefore *not* the counts the
    pre-spawn code produced from the same integer seed, so runs from before this
    fix are not reproduced value-for-value — do not mix old and new rows in one
    results table.
    """
    seed_sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed))
    count_seed, draw_seed = seed_sequence.spawn(2)
    return np.random.default_rng(count_seed), np.random.default_rng(draw_seed)


def inject(
    background: pd.DataFrame,
    pool: pd.DataFrame,
    lambda0: float,
    seed: int,
    *,
    days: list[str] | None = None,
    ratios: pd.DataFrame | None = None,
    mode: str = WINDOW_MODE,
    delta: float = SURGE_THRESHOLD,
    window: int = FEATURE_WINDOW,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build the modified corpus for one ``(lambda0, seed)``.

    Parameters
    ----------
    background : DataFrame
        Real background tweets, columns ``tweet_id, day``.
    pool : DataFrame
        Pool of relevant tweets, columns ``tweet_id, day`` (their original local
        posting date). Every day draws from the whole pool; ``day`` is kept for
        bookkeeping only and does not restrict the draws.
    lambda0 : float
        Mean number of injected tweets per day on a day the injection leaves at
        its baseline rate: ``lambda0`` exactly in window/step/ratio mode on a
        non-surge day, ``lambda0 * (1 + delta)`` on an elevated day.
    seed : int
        Seed of the injection. Child 0 of its SeedSequence drives the Poisson
        counts, child 1 the pool draws, so the two streams are independent.
    days, ratios : optional
        Precomputed reference days / surge ratios; computed on demand.
    mode, delta, window, horizon : optional
        Injection rate model, surge multiplier, spread length and the horizon the
        spread is offset by; see :func:`injection_counts`. ``window`` must match
        the window the features are built with (``paths.FEATURE_WINDOW``) and
        ``horizon`` the horizon the label was built with (``paths.HORIZON``) for
        the spread to land on the rows the model reads.

    Returns
    -------
    corpus : DataFrame
        Background rows plus injected rows, columns ``tweet_id, day, injected``
        (``injected`` is 1 for pool tweets, 0 for real background tweets).
    counts : DataFrame
        ``date, r, label, n_injected`` per day.
    info : dict
        ``lambda0``, ``seed``, ``mode``, ``delta``, ``window``, ``horizon``,
        ``n_background``, ``n_injected_total``, ``expected_total`` (= ``sum_t
        rate_t`` for the mode in use), ``mean_r``, ``n_elevated_days``,
        ``mean_injected_on_positive`` / ``..._negative`` days and the pool report
        (``pool_report``).
    """
    if days is None:
        days = sorted(background["day"].unique())
    if ratios is None:
        ratios = surge_ratio(days)
    ratios = ratios.set_index("date").reindex(days).reset_index()

    count_stream, draw_stream = injection_streams(seed)
    counts = injection_counts(ratios, lambda0, count_stream,
                              mode=mode, delta=delta, window=window,
                              horizon=horizon)
    draws = inject_pool_draws(pool, counts, draw_stream)

    bg = background[["tweet_id", "day"]].copy()
    bg["injected"] = 0
    draws["injected"] = 1
    corpus = pd.concat([bg, draws], ignore_index=True)

    table = ratios[["date", "r", "label"]].copy()
    table[INJECTED_COL] = counts.reindex(table["date"]).to_numpy()
    pos = table["label"] == 1
    elevated = surge_window_coverage(table["label"].to_numpy(), window, horizon) > 0
    info = {
        "lambda0": float(lambda0),
        "seed": int(seed),
        "mode": mode,
        "delta": float(delta),
        "window": int(window),
        "horizon": int(horizon),
        "n_background": int(len(bg)),
        "n_injected_total": int(len(draws)),
        "expected_total": expected_injections(
            table, lambda0, mode=mode, delta=delta, window=window, horizon=horizon
        ),
        "mean_r": float(table["r"].mean()),
        "n_elevated_days": int(elevated.sum()),
        "mean_injected_on_positive": float(table.loc[pos, INJECTED_COL].mean()),
        "mean_injected_on_negative": float(table.loc[~pos, INJECTED_COL].mean()),
        "pool": pool_report(pool, verbose=False),
    }
    return corpus, table, info


def main() -> None:
    from corpus import build_background, build_pool, load_llm_labels, load_unique_tweets

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lambda0", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=INJECTION_MODES, default=WINDOW_MODE)
    parser.add_argument("--delta", type=float, default=SURGE_THRESHOLD,
                        help="Surge multiplier of the rate, window/step modes.")
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW,
                        help="Days a surge spreads over; must match the feature window.")
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="Days the window mode's spread is offset ahead of the surge.")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    days = reference_days()
    tweets = load_unique_tweets()
    llm = load_llm_labels()
    background, pool = build_background(tweets, llm, days), build_pool(llm)
    ratios = surge_ratio(days)
    elevated = surge_window_coverage(ratios["label"].to_numpy(), args.window,
                                     args.horizon) > 0
    print(
        f"[inject] reference prevalence: {ratios['label'].mean():.3f} "
        f"({int(ratios['label'].sum())}/{len(ratios)} days), "
        f"mean r={ratios['r'].mean():.3f}"
    )
    if args.mode == WINDOW_MODE:
        print(
            f"[inject] mode={args.mode} window={args.window} horizon={args.horizon} "
            f"delta={args.delta:g}: "
            f"{int(elevated.sum())} elevated days ({100 * elevated.mean():.1f} % of days) "
            f"for a label positive on {100 * ratios['label'].mean():.1f} % of days "
            f"(the spread dilutes the contrast)."
        )

    for lam in args.lambda0:
        corpus, table, info = inject(
            background, pool, lam, args.seed, days=days, ratios=ratios,
            mode=args.mode, delta=args.delta, window=args.window,
            horizon=args.horizon,
        )
        out = (OUTPUTS_DIR / f"injection_{args.mode}_lambda{lam:g}_seed{args.seed}.csv")
        table.to_csv(out, index=False)
        print(
            f"[inject] lambda0={lam:g} seed={args.seed}: "
            f"n_injected={info['n_injected_total']} (expected {info['expected_total']:.1f}), "
            f"mean k on surge days={info['mean_injected_on_positive']:.2f}, "
            f"on other days={info['mean_injected_on_negative']:.2f} -> {out.name}"
        )


if __name__ == "__main__":
    main()
