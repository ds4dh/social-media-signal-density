"""Async client for the local vLLM server that reads out first-token logprobs.

The only backend is a Qwen model exposed through an OpenAI-compatible
``chat.completions`` endpoint. Requests pin ``temperature=0`` and
``max_tokens=1``, so a call is not a generation but a read-out of the model's
next-token distribution. That distribution is what turns a binary
``Yes``/``No`` question into a probability: :func:`answer_probabilities` sums
the probabilities of every token of the top-``k`` list that spells one of the
two answers, and :func:`p_informative` renormalises the two sums.

No tweet text leaves this module except towards the URL in ``BACKENDS``.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from openai import AsyncOpenAI

DEFAULT_BACKEND = "qwen3.8-flash-next"
DEFAULT_SEED = 42
DEFAULT_MAX_CONCURRENCY = 16
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60.0
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_CAP = 30.0

STATUS_OK = "ok"
STATUS_NO_ANSWER = "no_answer"
STATUS_ERROR = "error"

BACKENDS: dict[str, dict[str, Any]] = {
    "qwen3.8-flash-next": {
        "backend": "qwen_logprobs",
        "base_url": "https://ds4dh.unige.ch/vllm/v1",
        "model": "Qwen/Qwen3.8-Flash-Next-FP8",
        "api_key_env": "DS4DH_VLLM_API_KEY",
        "fingerprint": {
            "model": "Qwen/Qwen3.8-Flash-Next-FP8",
            "thinking": False,
            "top_logprobs": 20,
        },
    },
}


class LLMConfigError(RuntimeError):
    """Raised when the backend cannot be configured (unknown name, no key)."""


@dataclass(frozen=True)
class BackendConfig:
    """Resolved configuration of one backend."""

    name: str
    backend: str
    base_url: str
    model: str
    api_key_env: str
    fingerprint: dict[str, Any] = field(default_factory=dict)

    @property
    def top_logprobs(self) -> int:
        return int(self.fingerprint.get("top_logprobs", 20))

    @property
    def thinking(self) -> bool:
        return bool(self.fingerprint.get("thinking", False))

    def fingerprint_json(self) -> str:
        """Stable string form of the fingerprint, for storing next to results."""
        import json

        return json.dumps(self.fingerprint, sort_keys=True)


def get_backend(name: str = DEFAULT_BACKEND) -> BackendConfig:
    """Return the :class:`BackendConfig` registered under ``name``."""
    try:
        cfg = BACKENDS[name]
    except KeyError:
        known = ", ".join(sorted(BACKENDS))
        raise LLMConfigError(
            f"Unknown backend {name!r}. Known backends: {known}."
        ) from None
    return BackendConfig(
        name=name,
        backend=cfg["backend"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        api_key_env=cfg["api_key_env"],
        fingerprint=dict(cfg["fingerprint"]),
    )


def resolve_api_key(config: BackendConfig, api_key: str | None = None) -> str:
    """Read the API key from the environment, with a clear error if absent."""
    key = api_key if api_key is not None else os.environ.get(config.api_key_env)
    if not key:
        raise LLMConfigError(
            f"The environment variable {config.api_key_env} is not set. "
            f"It holds the API key of the local vLLM server at {config.base_url}. "
            f"Export it before running the classification, e.g. "
            f'export {config.api_key_env}="<key>".'
        )
    return key


def prompt_hash(template: str) -> str:
    """SHA-256 of the prompt template, used to version every stored label."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# From first-token logprobs to a probability
# ---------------------------------------------------------------------------

# Leading/trailing characters that a tokenizer may glue to the answer word.
_TRIM_CHARS = (
    string.punctuation
    + "\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00ab\u00bb\u00a0"
)
_ANSWER_WORDS = {"yes": "yes", "y": "yes", "no": "no", "n": "no"}


def normalize_answer_token(token: str | None) -> str | None:
    """Map a raw token to ``'yes'``/``'no'``, or ``None`` if it is neither.

    Whitespace, quotes, brackets and other punctuation around the word are
    ignored and case does not matter, so ``' Yes'``, ``'"Yes"'``, ``'YES.'``
    and ``'no!'`` all map to an answer.
    """
    if not isinstance(token, str):
        return None
    word = token.strip().strip(_TRIM_CHARS).strip().lower()
    return _ANSWER_WORDS.get(word)


def answer_probabilities(top_logprobs: Iterable[tuple[str | None, float]]) -> tuple[float, float]:
    """Sum the probabilities of the top tokens that spell Yes and No.

    ``top_logprobs`` is an iterable of ``(token, logprob)`` pairs for the first
    generated token. Returns ``(p_yes, p_no)``; both are ``0.0`` when the two
    answers are absent from the list.
    """
    p_yes = p_no = 0.0
    for token, logprob in top_logprobs:
        answer = normalize_answer_token(token)
        if answer is None:
            continue
        p = math.exp(logprob)
        if answer == "yes":
            p_yes += p
        else:
            p_no += p
    return p_yes, p_no


def p_informative(p_yes: float, p_no: float) -> tuple[float, str]:
    """Renormalise ``p_yes``/``p_no`` into ``p_informative`` plus a status."""
    total = p_yes + p_no
    if not total > 0.0:
        return math.nan, STATUS_NO_ANSWER
    return p_yes / total, STATUS_OK


def extract_first_token_logprobs(response: Any) -> list[tuple[str | None, float]]:
    """Pull ``(token, logprob)`` pairs of the first token out of an API reply."""
    try:
        choice = response.choices[0]
        content = choice.logprobs.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed API response: {exc}") from exc
    if not content:
        raise RuntimeError("API response carries no logprobs for the generated token.")
    top = content[0].top_logprobs or []
    pairs: list[tuple[str | None, float]] = []
    for entry in top:
        token = getattr(entry, "token", None)
        logprob = getattr(entry, "logprob", None)
        if token is None or logprob is None:
            continue
        pairs.append((token, float(logprob)))
    if not pairs:
        raise RuntimeError("API response carries an empty top_logprobs list.")
    return pairs


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff for the given 0-based retry attempt."""
    return min(base * (2 ** attempt), cap)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LogprobsClient:
    """Async OpenAI-compatible client returning first-token answer probabilities."""

    def __init__(
        self,
        config: BackendConfig | None = None,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_cap: float = DEFAULT_BACKOFF_CAP,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        sleep=asyncio.sleep,
    ) -> None:
        self.config = config if config is not None else get_backend()
        self.api_key = resolve_api_key(self.config, api_key)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.max_concurrency = max(1, int(max_concurrency))
        self._sleep = sleep
        self._client: AsyncOpenAI | None = None

    # -- low-level request ---------------------------------------------------

    def _sdk(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,  # retries are handled here, not in the SDK
            )
        return self._client

    async def _complete_once(self, prompt: str) -> list[tuple[str | None, float]]:
        """One request; returns the first-token top-logprob pairs."""
        response = await self._sdk().chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.config.top_logprobs,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.config.thinking}},
        )
        return extract_first_token_logprobs(response)

    # -- single prompt -------------------------------------------------------

    async def classify_prompt(self, prompt: str) -> dict[str, Any]:
        """Answer probabilities for one prompt, with retries on failure.

        The returned dict carries ``p_yes``, ``p_no``, ``p_informative``,
        ``status`` (``ok`` / ``no_answer`` / ``error``) and, for failures, the
        last error message.
        """
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                await self._sleep(_backoff_delay(attempt - 1, self.backoff_base, self.backoff_cap))
            try:
                top = await self._complete_once(prompt)
            except Exception as exc:  # network, timeout, malformed reply
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            p_yes, p_no = answer_probabilities(top)
            value, status = p_informative(p_yes, p_no)
            return {
                "p_yes": p_yes,
                "p_no": p_no,
                "p_informative": value,
                "status": status,
                "n_top_tokens": len(top),
                "error": "",
            }
        return {
            "p_yes": math.nan,
            "p_no": math.nan,
            "p_informative": math.nan,
            "status": STATUS_ERROR,
            "n_top_tokens": 0,
            "error": last_error,
        }

    # -- batch ---------------------------------------------------------------

    async def classify_many(self, prompts: Sequence[str]) -> list[dict[str, Any]]:
        """Classify ``prompts`` concurrently, capped at ``max_concurrency``."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _one(prompt: str) -> dict[str, Any]:
            async with semaphore:
                return await self.classify_prompt(prompt)

        if not prompts:
            return []
        return list(await asyncio.gather(*(_one(p) for p in prompts)))

    def classify_many_sync(self, prompts: Sequence[str]) -> list[dict[str, Any]]:
        """Synchronous wrapper around :meth:`classify_many`."""
        return asyncio.run(self.classify_many(prompts))

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
