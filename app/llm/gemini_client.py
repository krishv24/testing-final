from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 5
BASE_DELAY = 0.75
MAX_DELAY = 20.0


async def generate_json_text(prompt: str, *, temperature: float = 0.35) -> str:
    s = get_settings()
    if not s.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured")

    last_exc: Exception | None = None
    for attempt in range(DEFAULT_RETRIES):
        try:

            def _sync() -> str:
                client = genai.Client(api_key=s.gemini_api_key)
                resp = client.models.generate_content(
                    model=s.gemini_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )
                if not resp.text:
                    raise RuntimeError("Empty Gemini response")
                return resp.text

            return await asyncio.to_thread(_sync)
        except Exception as e:
            last_exc = e
            if attempt == DEFAULT_RETRIES - 1:
                break
            delay = min(MAX_DELAY, BASE_DELAY * (2**attempt))
            logger.warning("Gemini call failed (%s/%s): %s; retry in %.2fs", attempt + 1, DEFAULT_RETRIES, e, delay)
            await asyncio.sleep(delay)
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
