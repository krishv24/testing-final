from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.assistant.aggregation import aggregate_daily_from_hourly, aggregate_six_hour, merge_daily_tables
from app.assistant.climatology import fetch_climatology
from app.assistant.geocoding import resolve_location
from app.assistant.nws import maybe_attach_afd
from app.assistant.openmeteo import fetch_forecast
from app.cache_store import cache_get_json, cache_set_json
from app.config import ensure_cache_dir, get_settings
from app.degradation import DegradationCode
from app.schemas import AssistantRequest, ContextMode, ContextPayload

logger = logging.getLogger(__name__)


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _payload_hash(payload_dict: dict[str, Any]) -> str:
    raw = _canonical_json(payload_dict)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_lead_time_hours(
    hourly_timestamps: list[str],
    daily_dates: list[str],
) -> float:
    now = datetime.now(timezone.utc)
    latest: Optional[datetime] = None
    for ts in hourly_timestamps:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if latest is None or t > latest:
            latest = t
    for d in daily_dates:
        try:
            t = datetime.fromisoformat(d + "T23:59:59+00:00")
        except ValueError:
            continue
        if latest is None or t > latest:
            latest = t
    if latest is None:
        return 0.0
    return max(0.0, (latest - now).total_seconds() / 3600.0)


def trim_payload_for_mode(
    payload: ContextPayload,
    *,
    hierarchical_include_hourly: bool,
    baseline: bool,
) -> ContextPayload:
    p = payload.model_copy(deep=True)
    if baseline:
        p.six_hour_aggregates = []
        p.daily_aggregates = []
        return p
    if not hierarchical_include_hourly:
        p.hourly = []
    return p


async def run_assistant_pipeline(req: AssistantRequest, client: httpx.AsyncClient) -> tuple[ContextPayload, dict[str, Any]]:
    ensure_cache_dir()
    loc = await resolve_location(query=req.query, lat=req.latitude, lon=req.longitude, client=client)
    loc, afd_text, nws_codes = await maybe_attach_afd(loc, client)

    clim, clim_codes = await fetch_climatology(
        loc.latitude, loc.longitude, client, elevation_m=loc.elevation_m
    )
    current, hourly, daily_om, om_codes = await fetch_forecast(loc.latitude, loc.longitude, client)

    six = aggregate_six_hour(hourly)
    daily_h = aggregate_daily_from_hourly(hourly)
    daily_merged = merge_daily_tables(daily_h, daily_om)

    degradation = list(dict.fromkeys(clim_codes + om_codes + nws_codes))
    if not loc.wikipedia_summary:
        degradation.append(DegradationCode.WIKIPEDIA_UNAVAILABLE.value)

    lead_h = _compute_lead_time_hours([h.timestamp_utc for h in hourly], [d.date_utc for d in daily_merged])

    if req.context_style == "baseline":
        hierarchical_include_hourly = True
    else:
        hierarchical_include_hourly = lead_h < 7 * 24 + 1

    mode = ContextMode(
        name="baseline" if req.context_style == "baseline" else "hierarchical",
        hierarchical_include_hourly=hierarchical_include_hourly,
        lead_time_hours=lead_h,
    )

    payload = ContextPayload(
        context_mode=mode,
        location=loc,
        current_conditions=current,
        climatology=clim,
        hourly=hourly,
        six_hour_aggregates=six,
        daily_aggregates=daily_merged,
        nws_area_forecast_discussion=afd_text,
        degradation_codes=degradation,
        meta={
            "openmeteo_hourly_count": len(hourly),
            "daily_rows": len(daily_merged),
            "six_hour_rows": len(six),
            "realtime_source": "openmeteo_current",
            "current_observation_utc": current.timestamp_utc if current else None,
        },
    )

    trimmed = trim_payload_for_mode(
        payload,
        hierarchical_include_hourly=hierarchical_include_hourly,
        baseline=req.context_style == "baseline",
    )

    frozen = trimmed.model_dump(mode="json")
    cache_key = _payload_hash(frozen)

    if req.use_cache:
        cached = cache_get_json("context", cache_key)
        if cached:
            p2 = ContextPayload.model_validate(cached["payload"])
            p2.degradation_codes = list(dict.fromkeys(p2.degradation_codes + [DegradationCode.CACHE_HIT_CONTEXT.value]))
            return p2, {"cache_key": cache_key, "cached": True}

    cache_set_json("context", cache_key, {"payload": frozen, "assistant_request": req.model_dump()})
    return trimmed, {"cache_key": cache_key, "cached": False}


def build_context_payload(**kwargs: Any) -> ContextPayload:
    """Synchronous helper for tests: construct from keyword args matching ContextPayload."""
    return ContextPayload.model_validate(kwargs)
