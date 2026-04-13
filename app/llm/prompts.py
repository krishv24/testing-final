from __future__ import annotations

import json

from app.schemas import ContextPayload, ReportParams


def meteorologist_prompt(ctx: ContextPayload) -> str:
    payload = ctx.model_dump(mode="json")
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
{json.dumps(payload, indent=2, default=str)}

CONTEXT MODE: {mode_note}. {hourly_note}
REAL-TIME: {rt_note}

You must output a single JSON object with exactly these keys:
- "summary": string, several paragraphs. Describe weather dynamics across the full forecast horizon. Reference patterns from the daily table, the 6-hour table, and (if present) the hourly table. When `current_conditions` exists, relate it to the first hours of the forecast. Address short-term hourly dynamics, mesoscale 6-hour patterns, and daily persistent trends. Include causal reasoning (why conditions evolve, not only what they are). Keep the narrative internally consistent across time scales.
- "proof": string, a compact bullet-style or short-paragraph block listing observable data signals that justify the summary (specific numbers, trends, durations). Cover where applicable: pressure tendencies, wind speed changes, wind direction shifts, daily temperature amplitude, precipitation duration and intensity, humidity trends. Every claim in summary must be traceable here.
- "keywords": array of 3 to 5 short strings using controlled meteorological vocabulary (e.g. cooling trend, frontal passage, heavy rain, strong wind, unstable airmass, marine influence, overcast, light rain). Each keyword must correspond to at least one proof signal and at least one aggregate in the tables.
- "warnings": optional string. Include ONLY if data show hazardous or clearly anomalous conditions versus climatology (e.g. sustained winds far above normals, rainfall far above typical daily totals, icing risk, flooding risk from extreme multi-day rain). Each warning must cite data. Omit this key entirely if conditions are not extreme.

Rules:
- Rely primarily on aggregate values; do not over-weight single-hour spikes unless they persist.
- Separate daily trends from intraday variability when hourly data exist.
- For longer lead times without hourly data, focus on daily and 6-hour aggregates.
- Interpret sustained wind direction shifts as possible frontal or synoptic changes only when supported by temperature/humidity/precip patterns in the tables.
- Compare temperature and precipitation to climatology where possible to flag anomalies.
- Do not fabricate variables absent from the JSON (if pressure is missing, do not claim pressure values).

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
