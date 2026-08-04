from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from typing import Any

from google import genai

from app.config import get_settings

logger = logging.getLogger(__name__)

# Retry config — generous retries for transient 503/429 demand spikes
DEFAULT_RETRIES = 5
BASE_DELAY = 2.0   # seconds before first retry
MAX_DELAY = 60.0   # cap per-retry wait


def _is_transient_error(exc: Exception) -> bool:
    """Return True for errors that are safe to retry (503, 429, connection issues)."""
    msg = str(exc).lower()
    transient_markers = [
        "503",
        "unavailable",
        "429",
        "resource exhausted",
        "rate limit",
        "quota",
        "too many requests",
        "service unavailable",
        "temporarily unavailable",
        "high demand",
        "connection",
        "timeout",
    ]
    return any(m in msg for m in transient_markers)

# ── Global call counter so we can see exactly how many API hits happen ──
_CALL_COUNT = 0


def _get_caller_info() -> str:
    """Walk the stack to find who called generate_json_text (skip internals)."""
    for frame_info in inspect.stack():
        module = frame_info.filename
        if "gemini_client" in module:
            continue
        # Return the first non-gemini-client caller
        fn = frame_info.function
        lineno = frame_info.lineno
        # Shorten path for readability
        short = module.split("app\\")[-1] if "app\\" in module else module.split("app/")[-1] if "app/" in module else module
        return f"{short}:{fn}:{lineno}"
    return "unknown"


async def generate_json_text(prompt: str, *, temperature: float = 0.35) -> str:
    global _CALL_COUNT
    _CALL_COUNT += 1
    call_id = _CALL_COUNT

    s = get_settings()
    if not s.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    caller = _get_caller_info()
    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4  # rough estimate: 1 token ≈ 4 chars

    logger.info(
        "═══ GEMINI API CALL #%d ═══ caller=%s | model=%s | prompt=%d chars (~%d tokens) | temp=%.2f",
        call_id, caller, s.gemini_model, prompt_chars, prompt_tokens_est, temperature,
    )

    client = genai.Client(api_key=s.gemini_api_key)
    last_exc: Exception | None = None
    for attempt in range(DEFAULT_RETRIES):
        t0 = time.perf_counter()
        try:
            logger.info("  → Call #%d attempt %d/%d: sending request to Gemini...", call_id, attempt + 1, DEFAULT_RETRIES)
            resp = await client.aio.models.generate_content(
                model=s.gemini_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    max_output_tokens=4096,
                ),
            )
            elapsed = time.perf_counter() - t0
            resp_chars = len(resp.text) if resp.text else 0
            resp_tokens_est = resp_chars // 4

            # Try to extract usage metadata if the SDK provides it
            usage_info = ""
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                um = resp.usage_metadata
                usage_info = f" | usage: input={getattr(um, 'prompt_token_count', '?')} output={getattr(um, 'candidates_token_count', '?')} total={getattr(um, 'total_token_count', '?')}"

            logger.info(
                "  ✓ Call #%d SUCCESS in %.2fs | response=%d chars (~%d tokens)%s",
                call_id, elapsed, resp_chars, resp_tokens_est, usage_info,
            )

            if not resp.text:
                raise RuntimeError("Empty Gemini response")
            return resp.text
        except Exception as e:
            elapsed = time.perf_counter() - t0
            last_exc = e
            is_transient = _is_transient_error(e)
            logger.error(
                "  ✗ Call #%d FAILED attempt %d/%d in %.2fs | transient=%s | error_type=%s | error=%s",
                call_id, attempt + 1, DEFAULT_RETRIES, elapsed, is_transient, type(e).__name__, str(e)[:300],
            )
            # For non-transient errors (bad auth, bad request, etc.), fail immediately.
            if not is_transient:
                logger.error("  ✗ Call #%d non-transient error — aborting retries.", call_id)
                break
            if attempt == DEFAULT_RETRIES - 1:
                break
            # Exponential backoff: 2s, 4s, 8s, 16s, capped at MAX_DELAY
            delay = min(MAX_DELAY, BASE_DELAY * (2 ** attempt))
            logger.warning(
                "  … Call #%d got transient error (503/429). Retrying in %.1fs (attempt %d/%d)...",
                call_id, delay, attempt + 2, DEFAULT_RETRIES,
            )
            await asyncio.sleep(delay)

    logger.error(
        "═══ GEMINI CALL #%d FINAL FAILURE ═══ caller=%s | all %d attempts exhausted",
        call_id, caller, DEFAULT_RETRIES,
    )
    assert last_exc is not None
    raise last_exc


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)

