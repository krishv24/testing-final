from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.degradation import DegradationCode
from app.jsonutil import sha256_hex, canonical_json
from app.cache_store import cache_get_json, cache_set_json
from app.llm.gemini_client import generate_json_text, parse_json_object
from app.llm.prompts import writer_prompt
from app.schemas import (
    ContextPayload,
    FinalReport,
    MeteorologistOutput,
    ReportAnalysis,
    ReportContextEcho,
    ReportHeader,
    ReportParams,
)
from app.config import get_settings

logger = logging.getLogger(__name__)


class WriterLLMOutput(BaseModel):
    header: ReportHeader
    summary: str = Field(description="Adapted summary only")
    proof: str = Field(description="Proof with minimal edits")


async def run_writer(
    ctx: ContextPayload,
    meteorologist: MeteorologistOutput,
    params: ReportParams,
    *,
    use_cache: bool = True,
) -> tuple[FinalReport, dict[str, Any]]:
    m_dict = meteorologist.model_dump(mode="json")
    settings = get_settings()
    cache_payload = {
        "context": ctx.model_dump(mode="json"),
        "meteorologist": m_dict,
        "params": params.model_dump(mode="json"),
        "model": settings.gemini_model,
    }
    key = sha256_hex(cache_payload)
    if use_cache:
        hit = cache_get_json("report", key)
        if hit and "report" in hit:
            fr = FinalReport.model_validate(hit["report"])
            logger.info("Report cache hit key=%s", key[:16])
            return fr, {"cached": True, "cache_key": key, "degradation": [DegradationCode.CACHE_HIT_REPORT.value]}

    prompt = writer_prompt(m_dict, params, ctx)
    raw = await generate_json_text(prompt, temperature=0.5)
    data = parse_json_object(raw)
    wout = WriterLLMOutput.model_validate(data)

    log_mode = (
        f"{ctx.context_mode.name}; hourly_included={ctx.context_mode.hierarchical_include_hourly}; "
        f"lead_h={ctx.context_mode.lead_time_hours:.1f}"
    )

    ctx_echo = ReportContextEcho(
        mode="baseline" if ctx.context_mode.name == "baseline" else "hierarchical",
        daily=ctx.daily_aggregates,
        six_hour=ctx.six_hour_aggregates,
        hourly=ctx.hourly if ctx.context_mode.hierarchical_include_hourly else None,
        current_conditions=ctx.current_conditions,
        climatology=ctx.climatology,
        location=ctx.location,
    )

    analysis = ReportAnalysis(
        summary=wout.summary,
        proof=wout.proof if wout.proof.strip() else meteorologist.proof,
        keywords=list(meteorologist.keywords),
        warnings=meteorologist.warnings,
    )

    frozen = canonical_json(ctx.model_dump(mode="json"))
    report = FinalReport(
        header=wout.header,
        analysis=analysis,
        context=ctx_echo,
        context_frozen=frozen,
        meteorologist_model=settings.gemini_model,
        writer_model=settings.gemini_model,
        log_context_mode=log_mode,
    )

    cache_set_json(
        "report",
        key,
        {
            "report": report.model_dump(mode="json"),
            "frozen_inputs": cache_payload,
        },
    )
    return report, {"cached": False, "cache_key": key, "degradation": []}
