"""Prompt loading and tweet text preprocessing for the density estimation.

Preprocessing rules (Task.md section 1): drop URLs and user mentions, keep the
body of hashtags without the ``#``, then truncate to :data:`MAX_CHARS`. URLs are
removed before mentions and hashtags, because a URL can contain both an ``@``
and a ``#`` that are not mentions or hashtags.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT = "prompt"
MAX_CHARS = 1000

# http(s) URLs, bare www./t.co links, and @handles.
_URL_RE = re.compile(
    r"""(?:https?://|www\.)\S+ | (?<!\w)(?:\w+\.)?t\.co/\S+""",
    re.IGNORECASE | re.VERBOSE,
)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")


def load_prompt(name: str = DEFAULT_PROMPT) -> str:
    """Return the raw prompt template stored in ``prompts/<name>.txt``."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.txt")))
        raise FileNotFoundError(
            f"Prompt template not found: {path}. Available templates: {available}."
        )
    return path.read_text(encoding="utf-8")


def preprocess_tweet_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Clean one tweet body for the prompt.

    Returns the cleaned text with runs of whitespace collapsed and truncated to
    ``max_chars`` characters. Non-string input yields an empty string.
    """
    if not isinstance(text, str):
        return ""
    cleaned = _URL_RE.sub(" ", text)
    cleaned = _MENTION_RE.sub(" ", cleaned)
    cleaned = _HASHTAG_RE.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned[:max_chars]


def render_prompt(template: str, text: str, max_chars: int = MAX_CHARS) -> str:
    """Preprocess ``text`` and substitute it into the prompt template."""
    return template.format(text=preprocess_tweet_text(text, max_chars=max_chars))
