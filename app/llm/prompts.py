from __future__ import annotations

import json

from app.schemas import ContextPayload, ReportParams



def meteorologist_prompt(ctx: ContextPayload) -> str:
    payload = ctx.model_dump(mode="json")
    # Trim hourly to last 24 entries to save input tokens (aggregates cover the full period)
    if "hourly" in payload and len(payload["hourly"]) > 24:
        payload["hourly"] = payload["hourly"][-24:]
    mode_note = ctx.context_mode.name
    hourly_note = (
        "Full hourly series is included."
        if ctx.context_mode.hierarchical_include_hourly and ctx.hourly
        else "Hourly series omitted (lead time ≥ 7 days or token savings); rely on 6-hour and daily aggregates only."
    )
    rt_note = (
        "The field `current_conditions` (when present) is the latest near-real-time snapshot from the provider at `timestamp_utc` — anchor the present/now state to it before discussing forecast evolution."
        if ctx.current_conditions
        else "No separate current snapshot was provided; infer present conditions only from hourly/aggregates."
    )
    return f"""You are an expert meteorologist. You receive ONLY the structured JSON data below. Do not invent weather that is not supported by these tables.

DATA (JSON):
{json.dumps(payload, separators=(',', ':'), default=str)}

CONTEXT MODE: {mode_note}. {hourly_note}
REAL-TIME: {rt_note}

IMPORTANT: Keep your TOTAL response under 3000 tokens. Be concise.

Output a single JSON object with exactly these keys:
- "summary": string, 1-2 short paragraphs MAX. Describe key weather dynamics concisely with causal reasoning. Reference aggregate data. Keep it internally consistent.
- "proof": string, compact bullet list of key data signals (numbers, trends). Max 6-8 bullets.
- "keywords": array of 3-5 short meteorological terms (e.g. "cooling trend", "heavy rain").
- "warnings": optional string ONLY if hazardous conditions exist. Omit key if none.
- "claims": array of MAX 8 claim objects. Only the most important directional/threshold statements. Each: {{"claim_id":"CLAIM_001","variable":"...","assertion_type":"...","window_start_utc":"...","window_end_utc":"...","threshold_or_delta":"> 25.0","data_scale":"hourly|six_hour|daily|climatology|current"}}. threshold_or_delta MUST be a single numeric value with operator; never a range.
- "causal_chain": array of 4-8 strings ONLY from: pressure_drop, pressure_rise, stable_pressure, wind_increase, light_winds, wind_gusts, high_humidity, dry_air, rainfall, no_rain, warming, cooling, extreme_heat, low_visibility, fog_persistence, thunderstorm.
- "reasoning_flags": array of strings (uncertainties/anomalies, or empty list).

Rules: Use aggregates over single-hour spikes. Don't fabricate missing variables. Be concise.

Respond with JSON only, no markdown fences.
"""



def writer_prompt(meteorologist: dict, params: ReportParams, ctx: ContextPayload) -> str:
    m_json = json.dumps(meteorologist, indent=2, default=str)
    loc = ctx.location
    period = "the forecast period covered by the supplied tables"
    return f"""You are a professional weather communications editor.

METEOROLOGIST OUTPUT (facts; do not contradict):
{m_json}

USER PARAMETERS:
- tone: {params.tone}
- length: {params.length}
- application domain: {params.domain}

LOCATION: {loc.city}, {loc.administrative_region}, {loc.country} ({loc.latitude:.4f}, {loc.longitude:.4f})
FORECAST PERIOD NOTE: {period}

Task: Produce JSON with exactly:
- "header": {{ "title": string (location-aware report title), "information": string (short preamble about location and forecast period) }}
- "summary": string — adapt ONLY the meteorologist "summary" field for tone, length, and domain. Preserve all factual content; you may reorganize sentences but must not change numbers, directions, or timings implied by the meteorologist output.
- "proof": string — copy the meteorologist "proof" with minimal edits for readability and tone only; do not remove or alter quantitative facts.

Do NOT include keywords or warnings in your JSON (they will be merged from the meteorologist output server-side).

Respond with JSON only, no markdown fences.
"""
