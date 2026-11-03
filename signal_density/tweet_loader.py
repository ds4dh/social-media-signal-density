"""Load the tweet corpus for the density estimation.

The corpus is ``Rough_data/tweets.pkl`` (produced by
``Dataset construction/tweets_data.py``), truncated to tweets dated before
2023-01-01 — the window the dispatch-centre ground truth covers — and dated in
Europe/Zurich, like the rest of the pipeline. ``created_at`` is UTC, so the
local date of a tweet posted between 23:00 and midnight UTC moves to the
following day; the truncation itself is applied on that local date.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "Dataset construction"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from constants import DATA_DIR, LAT_G, LON_G  # noqa: E402

from signal_density import geo  # noqa: E402

LOCAL_TZ = "Europe/Zurich"
CUTOFF_LOCAL_DATE = pd.Timestamp("2023-01-01")
EARTH_RADIUS_KM = 6371.0

COLUMNS = ["id", "created_at", "lang", "text", "geo_place_id", "geo_coordinates.coordinates"]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two ``[lat, lon]`` points."""
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def load_tweets(
    pkl_path: Path | None = None,
    *,
    cutoff: pd.Timestamp = CUTOFF_LOCAL_DATE,
    max_distance_km: float = 350.0,
    places: dict[str, list[float]] | None = None,
    geometry=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return the corpus: ``tweet_id, created_at_utc, date_local, lang, text, in_geneva``.

    ``max_distance_km`` reproduces the filter of
    ``tweets_data.filter_tweets_by_location``: tweets with no usable location are
    kept (they come from the Swiss streams), while a located tweet is dropped
    when its point lies further than this from Geneva. ``verbose`` prints the
    number of tweets dropped by that filter.
    """
    pkl_path = Path(pkl_path) if pkl_path is not None else DATA_DIR / "tweets.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Tweet corpus not found: {pkl_path}")

    df = pd.read_pickle(pkl_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{pkl_path} is missing expected columns: {missing}")
    df = df[COLUMNS].copy()

    df["created_at_utc"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["date_local"] = df["created_at_utc"].dt.tz_convert(LOCAL_TZ).dt.normalize().dt.tz_localize(None)
    n_before = len(df)
    df = df[df["date_local"] < cutoff]
    if verbose:
        print(f"Corpus: {n_before} tweets, {len(df)} dated before {cutoff.date()} (Europe/Zurich).")

    if places is None:
        places = geo.place_bbox_map()
    points = geo.resolve_points(df, places)

    if max_distance_km is not None:
        # Same rule as tweets_data.filter_tweets_by_location: only a tweet with
        # its own coordinates can be too far away; a tweet that is merely
        # geo-tagged to a place is always kept.
        coords = df[geo.COORDS_COL].map(geo.tweet_point)
        far = coords.map(
            lambda p: p is not None and haversine_km(LAT_G, LON_G, p[1], p[0]) > max_distance_km
        )
        n_far = int(far.fillna(False).sum())
        df = df[~far.fillna(False)]
        points = points[~far.fillna(False)]
        if verbose:
            print(
                f"Location filter: dropped {n_far} tweets with coordinates more than "
                f"{max_distance_km:g} km from Geneva."
            )

    no_location = points.isna()
    if verbose:
        print(
            f"Location coverage: {int(no_location.sum())} of {len(df)} tweets "
            f"({100 * float(no_location.mean()):.1f}%) have no coordinates and no place "
            "bounding box; they are counted as not in Geneva."
        )

    if geometry is None:
        geometry = geo.load_canton_geometry()
    from shapely.geometry import Point
    from shapely.prepared import prep

    prepared = prep(geometry)
    df["in_geneva"] = points.map(
        lambda p: p is not None and prepared.contains(Point(p[0], p[1]))
    ).astype(bool)

    df = df.rename(columns={"id": "tweet_id"})
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["date_local"] = df["date_local"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)

SAMPLING_DESCRIPTION = (
    "drop duplicate tweet_id, sort by tweet_id, default_rng(seed).choice without replacement"
)


def random_sample(df: pd.DataFrame, n: int, seed: int, what: str = "tweets") -> pd.DataFrame:
    """Draw ``n`` distinct tweets without replacement, reproducibly.

    Duplicate ``tweet_id``s are collapsed first and the rest is sorted by
    ``tweet_id``, so the seed indexes a fixed order and the same corpus always
    yields the same draw whatever row order the loader returned. Output is
    ordered by ``(date_local, tweet_id)``.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    frame = df.drop_duplicates("tweet_id", keep="first")
    frame = frame.sort_values("tweet_id", kind="mergesort").reset_index(drop=True)
    if n > len(frame):
        raise ValueError(
            f"Cannot draw {n} {what} from {len(frame)} unique tweets. Lower --n or --sample."
        )
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(len(frame), size=n, replace=False))
    sample = frame.iloc[picked]
    return sample.sort_values(["date_local", "tweet_id"], kind="mergesort").reset_index(drop=True)
