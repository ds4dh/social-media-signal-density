from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from constants import *


def construct_textual_features(df, params, text_col="text", lang_col="lang"):
    """
    Extract entities from each tweet via fuzzy lexicon matching.

    Adds the following columns to df:
      - text_features_<entity>        : list of matched surface forms
      - text_features_<entity>_count  : integer count of matches
    """
    matchers = build_matchers(entity_lexicon=ENTITY_LEXICON)

    entity_results = {label: [] for label in ENTITY_LEXICON}
    entity_counts = {label: [] for label in ENTITY_LEXICON}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Lexicon pass"):
        text = row[text_col]
        lang = row[lang_col] if pd.notna(row[lang_col]) else "en"
        normalized = normalize_tweet(text)

        raw_matches = (
            matchers[lang].extract_keywords(normalized, span_info=True)
            if lang in matchers else []
        )
        kept = defaultdict(list)
        for entity_label, start, end in raw_matches:
            kept[entity_label].append(normalized[start:end])

        for label in ENTITY_LEXICON:
            entity_results[label].append(kept.get(label, []))
            entity_counts[label].append(len(kept.get(label, [])))

    for label in ENTITY_LEXICON:
        col = f"text_features_{label}"
        df[col] = entity_results[label]
        df[f"{col}_count"] = entity_counts[label]
    return df


def build_matchers(entity_lexicon):
    """
    Build one FuzzyKeywordProcessor per language.
    Each processor maps surface forms → entity label.
    """
    matchers = defaultdict(lambda: FuzzyKeywordProcessor(case_sensitive=False))
    for entity, lang_dict in entity_lexicon.items():
        for lang, surface_forms in lang_dict.items():
            for sf in surface_forms:
                matchers[lang].add_keyword(sf, entity)
    return dict(matchers)


def normalize_tweet(text):
    """
    Light normalization: strip URLs, user mentions, and excess whitespace.
    Hashtags are kept (without the #) since they often carry the entity.
    """
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_entities_lexicon(text, lang, matchers):
    """
    Returns a dict: {entity_label: [matched_surface_forms]}.
    """
    if lang not in matchers:
        return {}
    normalized = normalize_tweet(text)
    matches = matchers[lang].extract_keywords(normalized, span_info=True)
    entities = defaultdict(list)
    for entity_label, start, end in matches:
        entities[entity_label].append(normalized[start:end])
    return dict(entities)



def adaptive_cutoff(keyword):
    """
    Length-adaptive similarity threshold (0-100 scale, rapidfuzz convention).

    The threshold corresponds to allowing a fixed number of character-level
    edits at the most permissive end of each bin:
      - n <= 4  : 0 edits (exact match required)
      - 4 < n <= 8 : up to 1 edit
      - n > 8   : up to 2 edits

    Short keywords need exact matches to avoid false positives like
    'cough' matching 'tough'.
    """
    n = len(keyword.replace(" ", ""))
    if n <= 4:
        return 100
    elif n <= 8:
        return 80
    else:
        return 78


class FuzzyKeywordProcessor:
    """
    Drop-in replacement for flashtext.KeywordProcessor that supports
    fuzzy matching with per-keyword adaptive thresholds.
    """

    def __init__(self, case_sensitive=False):
        self.case_sensitive = case_sensitive
        self._index = []  # list of (entity_label, keyword, n_tokens, cutoff)

    def add_keyword(self, surface_form, entity_label):
        kw = surface_form if self.case_sensitive else surface_form.lower()
        n_tokens = len(kw.split())
        cutoff = adaptive_cutoff(kw)
        self._index.append((entity_label, kw, n_tokens, cutoff))

    def extract_keywords(self, text, span_info=True):
        """
        Returns list of (entity_label, start, end) if span_info=True,
        else list of entity_labels.
        Preserves the flashtext API.
        """
        if not text or not self._index:
            return []

        search_text = text if self.case_sensitive else text.lower()
        tokens = search_text.split()
        if not tokens:
            return []

        # Pre-compute character offsets for each token
        token_offsets = self._compute_token_offsets(search_text, tokens)

        raw_matches = []  # (entity, start, end, score, span_length)
        for entity, keyword, n_tok, cutoff in self._index:
            if n_tok > len(tokens):
                continue
            for i in range(len(tokens) - n_tok + 1):
                window = " ".join(tokens[i:i + n_tok])
                score = fuzz.ratio(keyword, window)
                if score >= cutoff:
                    char_start = token_offsets[i]
                    last_tok = tokens[i + n_tok - 1]
                    char_end = token_offsets[i + n_tok - 1] + len(last_tok)
                    raw_matches.append(
                        (entity, char_start, char_end, score, char_end - char_start)
                    )

        # Resolve overlaps: prefer longer span, then higher score
        resolved = self._resolve_overlaps(raw_matches)

        if span_info:
            return [(entity, start, end) for entity, start, end, _, _ in resolved]
        else:
            return [entity for entity, _, _, _, _ in resolved]

    @staticmethod
    def _compute_token_offsets(text, tokens):
        offsets = []
        pos = 0
        for tok in tokens:
            idx = text.find(tok, pos)
            offsets.append(idx)
            pos = idx + len(tok)
        return offsets

    @staticmethod
    def _resolve_overlaps(matches):
        """
        When multiple matches overlap (e.g., 'mask' inside 'mask usage'),
        keep the longest span. Ties broken by higher similarity score.
        """
        if not matches:
            return matches
        # Sort by span length descending, then by score descending
        sorted_matches = sorted(
            matches,
            key=lambda m: (m[4], m[3]),
            reverse=True,
        )
        kept = []
        for m in sorted_matches:
            entity, start, end, score, length = m
            overlaps = any(
                not (end <= ks or start >= ke)
                for _, ks, ke, _, _ in kept
            )
            if not overlaps:
                kept.append(m)
        # Re-sort by start position
        return sorted(kept, key=lambda m: m[1])

    def get_all_keywords(self):
        """Mirror flashtext API: return {surface_form: entity_label}."""
        return {kw: entity for entity, kw, _, _ in self._index}
