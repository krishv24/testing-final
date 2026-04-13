from __future__ import annotations

import logging

from app.degradation import DegradationCode
from app.jsonutil import sha256_hex
from app.cache_store import cache_get_json, cache_set_json
from app.config import get_settings
from app.llm.gemini_client import generate_json_text, parse_json_object
from app.llm.prompts import meteorologist_prompt
from app.schemas import ContextPayload, MeteorologistOutput
from app.validation import validate_for_meteorologist

logger = logging.getLogger(__name__)


async def run_meteorologist(ctx: ContextPayload, *, use_cache: bool = True) -> tuple[MeteorologistOutput, dict[str, object]]:
    validate_for_meteorologist(ctx)
    settings = get_settings()
    key = sha256_hex(
        {"context": ctx.model_dump(mode="json"), "model": settings.gemini_model}
    )
    if use_cache:
        hit = cache_get_json("meteorologist", key)
        if hit and "output" in hit:
            out = MeteorologistOutput.model_validate(hit["output"])
            logger.info("Meteorologist cache hit key=%s", key[:16])
            return out, {"cached": True, "cache_key": key, "degradation": [DegradationCode.CACHE_HIT_METEOROLOGIST.value]}

    prompt = meteorologist_prompt(ctx)
    raw = await generate_json_text(prompt, temperature=0.35)
    data = parse_json_object(raw)
    out = MeteorologistOutput.model_validate(data)

    cache_set_json(
        "meteorologist",
        key,
        {
            "output": out.model_dump(mode="json"),
            "context_hash": key,
        },
    )
    return out, {"cached": False, "cache_key": key, "degradation": []}
