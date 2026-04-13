"""Validate context payload before LLM stages."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from app.schemas import ContextPayload


def validate_context_dict(data: dict[str, Any]) -> ContextPayload:
    try:
        return ContextPayload.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid context payload: {e}") from e


def validate_for_meteorologist(ctx: ContextPayload) -> None:
    if not ctx.location or ctx.location.latitude is None or ctx.location.longitude is None:
        raise HTTPException(status_code=422, detail="context.location with lat/lon is required")
    if ctx.context_mode.name == "baseline":
        if not ctx.hourly:
            raise HTTPException(status_code=422, detail="baseline mode requires hourly series")
        return
    # hierarchical
    if not ctx.daily_aggregates:
        raise HTTPException(status_code=422, detail="hierarchical mode requires daily_aggregates")
    if not ctx.six_hour_aggregates:
        raise HTTPException(status_code=422, detail="hierarchical mode requires six_hour_aggregates")
    if ctx.context_mode.hierarchical_include_hourly and not ctx.hourly:
        raise HTTPException(status_code=422, detail="hierarchical mode with hourly requires hourly series")
