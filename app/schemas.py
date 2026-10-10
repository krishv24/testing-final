from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Location & climatology ---


class LocationMeta(BaseModel):
    city: str = ""
    administrative_region: str = ""
    country: str = ""
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    geoname_id: Optional[int] = None
    wikipedia_summary: Optional[str] = None
    nws_forecast_office: Optional[str] = None
    nws_grid_x: Optional[int] = None
    nws_grid_y: Optional[int] = None


class MonthlyClimateRow(BaseModel):
    month: int = Field(ge=1, le=12)
    tmin_c: Optional[float] = None
    tmax_c: Optional[float] = None
    ptot_mm: Optional[float] = None


class ClimatologyBlock(BaseModel):
    source: Literal["meteostat", "era5_open_meteo", "unavailable"] = "meteostat"
    station_id: Optional[str] = None
    monthly: list[MonthlyClimateRow] = Field(default_factory=list)


# --- Hourly & aggregates ---


class HourlyRow(BaseModel):
    timestamp_utc: str
    condition_code: int
    condition_label: str
    weather_category: str
    t_c: float
    t_feel_c: float
    td_c: float
    rh_percent: float
    wind_speed_ms: float
    wind_direction_deg: float
    wind_gust_ms: Optional[float] = None
    precipitation_mm: float = 0.0
    visibility_m: Optional[float] = None
    pressure_hpa: Optional[float] = None


class SixHourAggregateRow(BaseModel):
    window_start_utc: str
    window_end_utc: str
    t_mean_c: float
    t_min_c: float
    t_max_c: float
    rh_mean_percent: float
    wind_speed_mean_ms: float
    wind_direction_mean_deg: Optional[float] = None
    precipitation_sum_mm: float
    td_mean_c: float
    visibility_mean_m: Optional[float] = None
    pressure_mean_hpa: Optional[float] = None


class DailyAggregateRow(BaseModel):
    date_utc: str
    aggregate_source: Literal["hourly_derived", "open_meteo_daily_forecast"] = "hourly_derived"
    t_mean_c: float
    t_min_c: float
    t_max_c: float
    rh_mean_percent: float
    wind_speed_mean_ms: float
    precipitation_sum_mm: float
    td_mean_c: float
    visibility_mean_m: Optional[float] = None
    pressure_mean_hpa: Optional[float] = None


# --- Context payload (Block 1 output) ---


class ContextMode(BaseModel):
    name: Literal["baseline", "hierarchical"] = "hierarchical"
    hierarchical_include_hourly: bool = True
    lead_time_hours: float = 0.0


class ContextPayload(BaseModel):
    version: str = "1.0"
    context_mode: ContextMode
    location: LocationMeta
    current_conditions: Optional[HourlyRow] = Field(
        default=None,
        description="Open-Meteo current — latest conditions at API request time (near real-time).",
    )
    climatology: ClimatologyBlock
    hourly: list[HourlyRow] = Field(default_factory=list)
    six_hour_aggregates: list[SixHourAggregateRow] = Field(default_factory=list)
    daily_aggregates: list[DailyAggregateRow] = Field(default_factory=list)
    nws_area_forecast_discussion: Optional[str] = None
    degradation_codes: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


# --- Meteorologist (Block 2) ---


class WeatherClaim(BaseModel):
    claim_id: str = Field(description="Unique claim identifier (e.g., CLAIM_001)")
    variable: str = Field(description="Name of the meteorological variable (e.g., temperature, wind_speed, precipitation)")
    assertion_type: str = Field(description="Type of assertion (e.g., trend, threshold_exceeded, comparison_to_normal)")
    window_start_utc: str = Field(description="ISO timestamp of the start of the time window for the claim")
    window_end_utc: str = Field(description="ISO timestamp of the end of the time window for the claim")
    threshold_or_delta: str = Field(description="Numerical threshold or expected delta/change (e.g., > 25.0, +5.0)")
    data_scale: Literal["hourly", "six_hour", "daily", "climatology", "current"] = Field(
        description="Data scale or table from which this claim was derived"
    )

class MeteorologistOutput(BaseModel):
    summary: str
    proof: str
    keywords: list[str] = Field(default_factory=list, min_length=3)
    warnings: Optional[str] = None
    claims: list[WeatherClaim] = Field(default_factory=list)
    causal_chain: list[str] = Field(default_factory=list)
    reasoning_flags: list[str] = Field(default_factory=list)
    confidence_report: Optional[dict[str, Any]] = None

    @field_validator("proof", "summary", mode="before")
    @classmethod
    def coerce_to_string(cls, v):
        """LLMs sometimes return these as a list of strings; join them."""
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v

    @field_validator("keywords", mode="before")
    @classmethod
    def truncate_keywords(cls, v: list[str]) -> list[str]:
        """LLMs occasionally return >5 keywords; silently truncate to 5."""
        if isinstance(v, list) and len(v) > 5:
            return v[:5]
        return v



# --- Writer (Block 3) ---


Tone = Literal["technical", "conversational", "official"]
Length = Literal["short", "medium", "long"]
Domain = Literal["risk_analysis", "energy", "extreme_weather", "urban_planning", "agriculture", "general_public"]


class ReportParams(BaseModel):
    tone: Tone = "conversational"
    length: Length = "medium"
    domain: Domain = "general_public"


class ReportHeader(BaseModel):
    title: str
    information: str


class ReportAnalysis(BaseModel):
    summary: str
    proof: str
    keywords: list[str]
    warnings: Optional[str] = None


class ReportContextEcho(BaseModel):
    mode: Literal["baseline", "hierarchical"]
    daily: list[DailyAggregateRow]
    six_hour: list[SixHourAggregateRow]
    hourly: Optional[list[HourlyRow]] = None
    current_conditions: Optional[HourlyRow] = None
    climatology: ClimatologyBlock
    location: LocationMeta


class FinalReport(BaseModel):
    header: ReportHeader
    analysis: ReportAnalysis
    context: ReportContextEcho
    context_frozen: str = Field(description="Exact JSON snapshot of inputs for reproducibility")
    meteorologist_model: str = ""
    writer_model: str = ""
    log_context_mode: str = ""


def parse_lat_lon_string(text: str) -> Optional[tuple[float, float]]:
    """Parse coordinate pairs from text like '19.076, 72.877' or '19.076° N, 72.877° E'."""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    m = re.match(r"^([-+]?[0-9]*\.?[0-9]+)[,\s/]+([-+]?[0-9]*\.?[0-9]+)$", s)
    if m:
        try:
            lat = float(m.group(1))
            lon = float(m.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass
    m = re.match(
        r"^([0-9]*\.?[0-9]+)\s*°?\s*([NSns])[,\s/]+([0-9]*\.?[0-9]+)\s*°?\s*([EWew])$",
        s,
    )
    if m:
        try:
            lat = float(m.group(1)) * (-1 if m.group(2).upper() == "S" else 1)
            lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass
    return None


def normalize_location_data(data: Any) -> Any:
    """Normalize input dict so lat/lon aliases, empty query strings, and coordinates are properly routed."""
    if not isinstance(data, dict):
        return data

    lat_val = data.get("latitude")
    if lat_val is None or lat_val == "":
        for alias in ("lat", "Latitude", "Lat"):
            if alias in data and data[alias] not in (None, ""):
                lat_val = data[alias]
                break

    lon_val = data.get("longitude")
    if lon_val is None or lon_val == "":
        for alias in ("lon", "long", "lng", "Longitude", "Lon", "Long", "Lng"):
            if alias in data and data[alias] not in (None, ""):
                lon_val = data[alias]
                break

    if isinstance(lat_val, str) and (lon_val is None or lon_val == ""):
        parsed = parse_lat_lon_string(lat_val)
        if parsed:
            lat_val, lon_val = parsed

    q = data.get("query")
    if isinstance(q, str):
        q = q.strip()
        if not q:
            q = None
        elif (lat_val is None or lat_val == "") and (lon_val is None or lon_val == ""):
            parsed = parse_lat_lon_string(q)
            if parsed:
                lat_val, lon_val = parsed
                q = None

    if isinstance(lat_val, str) and lat_val.strip():
        try:
            lat_val = float(lat_val.strip())
        except ValueError:
            pass
    elif lat_val == "":
        lat_val = None

    if isinstance(lon_val, str) and lon_val.strip():
        try:
            lon_val = float(lon_val.strip())
        except ValueError:
            pass
    elif lon_val == "":
        lon_val = None

    data["latitude"] = lat_val
    data["longitude"] = lon_val
    data["query"] = q
    return data


# --- API bodies ---


class AssistantRequest(BaseModel):
    query: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    context_style: Literal["hierarchical", "baseline"] = "hierarchical"
    use_cache: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, data: Any) -> Any:
        return normalize_location_data(data)


class AnalysisRequest(BaseModel):
    context: dict[str, Any]
    use_cache: bool = True


class ReportRequest(BaseModel):
    context: dict[str, Any]
    tone: Tone = "conversational"
    length: Length = "medium"
    domain: Domain = "general_public"
    use_cache: bool = True


class FullPipelineRequest(BaseModel):
    """One-shot: Block 1 → Block 2 → Block 3 (for UI / demos)."""

    query: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    context_style: Literal["hierarchical", "baseline"] = "hierarchical"
    tone: Tone = "conversational"
    length: Length = "medium"
    domain: Domain = "general_public"
    use_cache: bool = True

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, data: Any) -> Any:
        return normalize_location_data(data)
