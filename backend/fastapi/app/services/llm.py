"""Single seam for LLM text generation.

Why a seam
----------
Free-tier availability and model names change often. Every LLM call in the
project goes through :func:`generate`, so switching provider or model is a
one-function change rather than a hunt through the codebase.

Privacy rule (non-negotiable)
-----------------------------
**No image bytes and no patient identifiers are ever sent to this function.**
Callers pass only the structured classification result and passages already
retrieved from the local knowledge base. The scan never leaves the machine.
That is both a safety property worth stating in the write-up and the reason
payloads stay tiny enough that free-tier limits are irrelevant.

Usage context
-------------
This is an **offline authoring aid**, not a request-time dependency. The
narrative output space is small and enumerable (4 classes x 3 uncertainty
bands = 12), so narratives are generated once, reviewed by a human, and
committed. Serving reads the vetted text and never calls an LLM, which means
zero runtime latency, no rate limits during a demo, and every clinical sentence
having been read by a person before a user sees it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.2

# Generous by necessity. The gpt-oss family are reasoning models: internal
# reasoning is billed against the completion budget before any visible content
# is produced. A 16-token cap returned an empty message with
# completion_tokens=16 -- the reasoning had consumed the entire allowance. Leave
# room for reasoning plus the answer.
DEFAULT_MAX_TOKENS = 3000

# Retry policy for transient failures and rate limits. The free tier allows far
# more than this project needs, so a short backoff is sufficient.
MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 2.0


@dataclass
class LLMResult:
    """Outcome of one generation attempt."""

    text: str = ""
    model: str = ""
    provider: str = "groq"
    ok: bool = False
    error: str | None = None
    usage: dict = field(default_factory=dict)
    attempts: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "ok": self.ok,
            "error": self.error,
            "usage": self.usage,
            "attempts": self.attempts,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def is_configured() -> bool:
    """Whether an API key is present.

    Checked before use so a missing key produces a clear message rather than an
    authentication error from deep inside the client library.
    """
    return bool(settings.GROQ_API)


def _client():
    """Construct the Groq client.

    Raises:
        RuntimeError: No API key configured, or the package is missing.
    """
    if not is_configured():
        raise RuntimeError(
            "GROQ_API is not set. Add it to the repo-root .env "
            "(see .env.example) before running narrative authoring."
        )
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The 'groq' package is not installed. pip install groq") from exc

    return Groq(api_key=settings.GROQ_API)


def generate(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> LLMResult:
    """Generate text from a prompt.

    Args:
        prompt: User message. Must contain only non-identifying structured data
            and locally retrieved passages.
        system: Optional system message constraining the model's behaviour.
        model: Override ``settings.GROQ_MODEL``.
        temperature: Low by default — this is factual summarisation constrained
            to supplied evidence, not creative writing.
        max_tokens: Response cap.

    Returns:
        An :class:`LLMResult`. Failures are returned rather than raised, so a
        batch authoring run records which items failed and continues.
    """
    model_name = model or settings.GROQ_MODEL
    started = time.time()
    result = LLMResult(model=model_name)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        client = _client()
    except RuntimeError as exc:
        result.error = str(exc)
        result.elapsed_seconds = time.time() - started
        return result

    last_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result.text = (response.choices[0].message.content or "").strip()
            result.ok = bool(result.text)
            if response.usage is not None:
                result.usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", None
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            if not result.ok:
                completion = result.usage.get("completion_tokens")
                if completion and completion >= max_tokens:
                    result.error = (
                        f"Empty message with completion_tokens={completion} at "
                        f"max_tokens={max_tokens}. Reasoning models consume the "
                        "completion budget before emitting content — raise "
                        "max_tokens."
                    )
                else:
                    result.error = "Provider returned an empty message."
            result.elapsed_seconds = time.time() - started
            return result

        except Exception as exc:  # noqa: BLE001 - provider errors vary by type
            last_error = f"{type(exc).__name__}: {exc}"
            transient = any(
                marker in last_error.lower()
                for marker in ("rate", "429", "timeout", "503", "502", "connection")
            )
            if attempt < MAX_ATTEMPTS and transient:
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "LLM attempt %d/%d failed (%s). Retrying in %.0fs.",
                    attempt,
                    MAX_ATTEMPTS,
                    last_error[:120],
                    wait,
                )
                time.sleep(wait)
                continue

            # Non-transient, or out of attempts.
            break

    result.error = last_error
    result.elapsed_seconds = time.time() - started
    return result


def health() -> dict:
    """Cheap configuration report that never exposes the key."""
    return {
        "provider": "groq",
        "configured": is_configured(),
        "model": settings.GROQ_MODEL,
        "key_length": len(settings.GROQ_API) if settings.GROQ_API else 0,
    }
