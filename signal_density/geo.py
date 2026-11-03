"""Decide whether a tweet is located inside the canton of Geneva.

``tweets.pkl`` carries two location fields: ``geo_coordinates.coordinates``
(a ``[lon, lat]`` pair, present for ~5.6 % of tweets) and ``geo_place_id``
(present for ~97 % of tweets). The place id is only a hash — the bounding box
it refers to lives in the raw twarc dumps under ``Rough_data/*.jsonl``, in the
``includes.places`` entries, and those files are read-only inputs here.

Polygon source
--------------
The canton boundary is the OpenStreetMap administrative boundary of the
canton of Geneva (OSM relation 1702419, ``admin_level=4``), retrieved from
Nominatim on 2026-10-28 and cached in ``data/geneva_canton_osm.json`` so that
runs are reproducible and offline. Coordinates are WGS84 ``[lon, lat]``, the
same order the Twitter API uses. © OpenStreetMap contributors (ODbL 1.0).

Decision rule
-------------
* a tweet with coordinates is tested as that point;
* otherwise the tweet's place bounding box is reduced to its centroid and that
  centroid is tested — Twitter only exposes the box, so the centroid is the
  single best point estimate available;
* a tweet with neither is reported as *not* in Geneva. Such tweets are counted
  and printed, because they make :data:`in_geneva` conservative (an
  under-count of Geneva density rather than an inflation).

The test is a point-in-polygon test, so a bounding-box centroid that happens to
fall just outside the canton (for example a French suburb whose city polygon
straddles the border) is excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely.geometry import MultiPolygon, Point, shape
from shapely.prepared import prep

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_TWEETS_DIR = REPO_ROOT / "Dataset construction" / "Rough_data"
POLYGON_FILE = DATA_DIR / "geneva_canton_osm.json"
PLACE_CACHE_FILE = DATA_DIR / "place_bboxes.json"

COORDS_COL = "geo_coordinates.coordinates"
PLACE_ID_COL = "geo_place_id"


class LocationDataError(RuntimeError):
    """Raised when the location inputs needed by :func:`in_geneva` are unusable."""


def load_canton_geometry(polygon_file: Path = POLYGON_FILE):
    """Load the cached canton-of-Geneva polygon as a shapely geometry."""
    if not polygon_file.exists():
        raise LocationDataError(
            f"Missing canton polygon {polygon_file}. It must hold the OSM "
            "boundary of the canton of Geneva (relation 1702419)."
        )
    payload = json.loads(polygon_file.read_text(encoding="utf-8"))
    geometry = shape(payload["features"][0]["geometry"])
    if geometry.is_empty:
        raise LocationDataError(f"Canton polygon in {polygon_file} is empty.")
    return geometry


def build_place_bbox_map(raw_dir: Path = RAW_TWEETS_DIR) -> dict[str, list[float]]:
    """Map every Twitter ``place_id`` to its ``[min_lon, min_lat, max_lon, max_lat]``.

    Scans the read-only twarc dumps for ``includes.places`` entries. Duplicate
    ids agree on their bbox in this corpus, so the first sighting wins.
    """
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise LocationDataError(f"No raw '.jsonl' dumps found in {raw_dir}.")
    places: dict[str, list[float]] = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                for place in (payload.get("includes") or {}).get("places") or []:
                    place_id = place.get("id")
                    bbox = (place.get("geo") or {}).get("bbox")
                    if place_id and bbox and len(bbox) == 4:
                        places.setdefault(place_id, list(bbox))
    if not places:
        raise LocationDataError(f"No place bounding boxes found in {raw_dir}.")
    return places


def place_bbox_map(
    raw_dir: Path = RAW_TWEETS_DIR,
    cache_file: Path = PLACE_CACHE_FILE,
    rebuild: bool = False,
) -> dict[str, list[float]]:
    """Return the place-id → bbox map, building the JSON cache on first use."""
    if cache_file.exists() and not rebuild:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        places = cached.get("places") or {}
        if places:
            return places
    places = build_place_bbox_map(raw_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "source": "includes.places of the read-only twarc dumps in "
                "'Dataset construction/Rough_data/*.jsonl'",
                "bbox_format": "[min_lon, min_lat, max_lon, max_lat], WGS84",
                "n_places": len(places),
                "places": places,
            }
        ),
        encoding="utf-8",
    )
    return places


def tweet_point(value) -> tuple[float, float] | None:
    """Return ``(lon, lat)`` from a ``geo_coordinates.coordinates`` value."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lon, lat = value
        if lon is not None and lat is not None:
            return float(lon), float(lat)
    return None


def bbox_centroid(bbox) -> tuple[float, float] | None:
    """Return the centroid ``(lon, lat)`` of a ``[min_lon, min_lat, max_lon, max_lat]`` box."""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
        return ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)
    return None


def resolve_points(df: pd.DataFrame, places: dict[str, list[float]]) -> pd.Series:
    """Best available location point per tweet: coordinates, else place centroid.

    Returns an object Series holding ``(lon, lat)`` tuples and ``None`` where
    the corpus carries no usable location.
    """
    if COORDS_COL not in df.columns or PLACE_ID_COL not in df.columns:
        raise LocationDataError(
            f"Expected columns {COORDS_COL!r} and {PLACE_ID_COL!r} to be present."
        )

    def _resolve(row):
        point = tweet_point(row[COORDS_COL])
        if point is not None:
            return point
        place_id = row[PLACE_ID_COL]
        if isinstance(place_id, str):
            return bbox_centroid(places.get(place_id))
        return None

    return df.apply(_resolve, axis=1)


def in_geneva(
    df: pd.DataFrame,
    places: dict[str, list[float]] | None = None,
    geometry=None,
) -> pd.Series:
    """Boolean Series: True when the tweet's location falls inside the canton.

    Tweets without a usable location are ``False``; :func:`location_coverage`
    reports how many those are.
    """
    if places is None:
        places = place_bbox_map()
    if geometry is None:
        geometry = load_canton_geometry()

    points = resolve_points(df, places)
    prepared = prep(geometry)
    mask = points.map(
        lambda p: p is not None and prepared.contains(Point(p[0], p[1]))
    )
    return mask.astype(bool).rename("in_geneva")


def location_coverage(df: pd.DataFrame, places: dict[str, list[float]]) -> pd.Series:
    """Boolean Series: True when a tweet has any usable location at all."""
    return resolve_points(df, places).map(lambda p: p is not None).rename("has_location")
