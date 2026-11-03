"""Shared paths and fixed experiment settings for the semi-synthetic experiment.

Importing this module puts ``Dataset construction/``, ``Models/`` and the
repository root on ``sys.path`` so that the experiment can reuse the real
pipeline code by importing it (Task.md: reuse, do not fork).
"""

from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

DATASET_CONSTRUCTION_DIR = REPO_ROOT / "Dataset construction"
MODELS_DIR = REPO_ROOT / "Models"
SIGNAL_DENSITY_DIR = REPO_ROOT / "signal_density"

TWEETS_PKL = DATASET_CONSTRUCTION_DIR / "Rough_data" / "tweets.pkl"
FINAL_DATASET_DIR = DATASET_CONSTRUCTION_DIR / "FINAL_DATASET"
#: full-corpus LLM run, one row per unique tweet. ``date_local`` (the
#: Europe/Zurich posting date) is carried for bookkeeping and reporting only:
#: the injection draws every day from the whole pool and ignores the posting date.
LLM_LABELS_PARQUET = SIGNAL_DENSITY_DIR / "outputs" / "llm_labels.parquet"
LLM_SCORES_CSV = SIGNAL_DENSITY_DIR / "outputs" / "tweet_scores.csv"
LLM_LABELS_PARAMS_JSON = SIGNAL_DENSITY_DIR / "outputs" / "llm_labels_params.json"
DAILY_COUNTS_CSV = SIGNAL_DENSITY_DIR / "outputs" / "tweet_scores.csv"

CACHE_DIR = EXP_DIR / "cache"
OUTPUTS_DIR = EXP_DIR / "outputs"

LEXICON_CACHE = CACHE_DIR / "lexicon.parquet"
LLM_CACHE = CACHE_DIR / "llm.parquet"
BERT_MATRIX = CACHE_DIR / "bert.npy"
BERT_IDS = CACHE_DIR / "bert_ids.parquet"

#: reference dataset produced by the real pipeline (``FINAL_DATASET/dataset_<name>.pkl``)
REFERENCE_DATASET = "matching_horizon_3"
REFERENCE_DATASET_PKL = FINAL_DATASET_DIR / f"dataset_{REFERENCE_DATASET}.pkl"


HORIZON = 3
SURGE_THRESHOLD = 0.2
LABELING_STRATEGY = "surge_detection"
DISPATCH_UNIT = "patients"
TARGET_SCOPE = "respiratory"


FEATURE_WINDOW = 1

#: row-based lags implementing ``FEATURE_WINDOW`` (lag 0 is the day itself)
FEATURE_WINDOW_LAGS = tuple(range(1, FEATURE_WINDOW))

# Decision 1 / 3: only these two methods, with the real pipeline's settings.
BERT_MODEL_NAME = "Twitter/twhin-bert-base"
BERT_MAX_LENGTH = 512
BERT_BATCH_SIZE = 32

#: expected prompt hash of the full-corpus LLM run (Task.md step 1)
LLM_PROMPT_HASH_PREFIX = "a6224977"

LOCAL_TZ = "Europe/Zurich"


def add_repo_paths() -> None:
    """Make the real pipeline modules importable (idempotent)."""
    for path in (str(REPO_ROOT), str(DATASET_CONSTRUCTION_DIR), str(MODELS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


add_repo_paths()
