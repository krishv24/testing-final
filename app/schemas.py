from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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


# --- API bodies ---


class AssistantRequest(BaseModel):
    query: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    context_style: Literal["hierarchical", "baseline"] = "hierarchical"
    use_cache: bool = True


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
