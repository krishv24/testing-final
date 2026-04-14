from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.assistant.categories import category_from_wmo_code, label_from_wmo_code
from app.assistant.retry import with_exponential_backoff
from app.degradation import DegradationCode
from app.schemas import DailyAggregateRow, HourlyRow

logger = logging.getLogger(__name__)


def _hourly_row(data: dict[str, list[Any]], idx: int) -> HourlyRow:
    timestamp_iso = str(data["time"][idx])
    if not timestamp_iso.endswith("Z") and "+" not in timestamp_iso:
        timestamp_iso += "Z"

    code = int(data["weather_code"][idx] if data["weather_code"][idx] is not None else 0)
    label = label_from_wmo_code(code)
    precip = float(data["precipitation"][idx] if data["precipitation"][idx] is not None else 0.0)
    cat = category_from_wmo_code(code, precip)

    return HourlyRow(
        timestamp_utc=timestamp_iso,
        condition_code=code,
        condition_label=label,
        weather_category=cat,
        t_c=float(data["temperature_2m"][idx] if data["temperature_2m"][idx] is not None else 0.0),
        t_feel_c=float(data["apparent_temperature"][idx] if data["apparent_temperature"][idx] is not None else (data["temperature_2m"][idx] or 0.0)),
        td_c=float(data["dew_point_2m"][idx] if data["dew_point_2m"][idx] is not None else 0.0),
        rh_percent=float(data["relative_humidity_2m"][idx] if data["relative_humidity_2m"][idx] is not None else 0.0),
        wind_speed_ms=float(data["wind_speed_10m"][idx] if data["wind_speed_10m"][idx] is not None else 0.0),
        wind_direction_deg=float(data["wind_direction_10m"][idx] if data["wind_direction_10m"][idx] is not None else 0.0),
        wind_gust_ms=float(data["wind_gusts_10m"][idx]) if data.get("wind_gusts_10m") and data["wind_gusts_10m"][idx] is not None else None,
        precipitation_mm=precip,
        visibility_m=float(data["visibility"][idx]) if data.get("visibility") and data["visibility"][idx] is not None else None,
        pressure_hpa=float(data["surface_pressure"][idx]) if data.get("surface_pressure") and data["surface_pressure"][idx] is not None else None,
    )


def _current_row(data: dict[str, Any]) -> HourlyRow:
    timestamp_iso = str(data["time"])
    if not timestamp_iso.endswith("Z") and "+" not in timestamp_iso:
        timestamp_iso += "Z"

    code = int(data.get("weather_code", 0) if data.get("weather_code") is not None else 0)
    label = label_from_wmo_code(code)
    precip = float(data.get("precipitation", 0.0) if data.get("precipitation") is not None else 0.0)
    cat = category_from_wmo_code(code, precip)

    return HourlyRow(
        timestamp_utc=timestamp_iso,
        condition_code=code,
        condition_label=label,
        weather_category=cat,
        t_c=float(data.get("temperature_2m", 0.0) if data.get("temperature_2m") is not None else 0.0),
        t_feel_c=float(data.get("apparent_temperature", data.get("temperature_2m", 0.0)) if data.get("apparent_temperature") is not None else 0.0),
        td_c=float(data.get("dew_point_2m", 0.0) if data.get("dew_point_2m") is not None else 0.0),
        rh_percent=float(data.get("relative_humidity_2m", 0.0) if data.get("relative_humidity_2m") is not None else 0.0),
        wind_speed_ms=float(data.get("wind_speed_10m", 0.0) if data.get("wind_speed_10m") is not None else 0.0),
        wind_direction_deg=float(data.get("wind_direction_10m", 0.0) if data.get("wind_direction_10m") is not None else 0.0),
        wind_gust_ms=float(data["wind_gusts_10m"]) if data.get("wind_gusts_10m") is not None else None,
        precipitation_mm=precip,
        visibility_m=float(data["visibility"]) if data.get("visibility") is not None else None,
        pressure_hpa=float(data["surface_pressure"]) if data.get("surface_pressure") is not None else None,
    )


def _daily_row(data: dict[str, list[Any]], idx: int) -> DailyAggregateRow:
    date_iso = str(data["time"][idx])

    tmax = float(data["temperature_2m_max"][idx] if data["temperature_2m_max"][idx] is not None else 0.0)
    tmin = float(data["temperature_2m_min"][idx] if data["temperature_2m_min"][idx] is not None else 0.0)
    tmean = (tmax + tmin) / 2.0

    return DailyAggregateRow(
        date_utc=date_iso,
        aggregate_source="open_meteo_daily_forecast",
        t_mean_c=tmean,
        t_min_c=tmin,
        t_max_c=tmax,
        rh_mean_percent=0.0,
        wind_speed_mean_ms=float(data["wind_speed_10m_max"][idx] if data["wind_speed_10m_max"][idx] is not None else 0.0) / 2.0, # Approximate
        precipitation_sum_mm=float(data["precipitation_sum"][idx] if data["precipitation_sum"][idx] is not None else 0.0),
        td_mean_c=0.0,
        visibility_mean_m=None,
        pressure_mean_hpa=float(data["surface_pressure_mean"][idx]) if data.get("surface_pressure_mean") and data["surface_pressure_mean"][idx] is not None else None,
    )


async def fetch_forecast(
    lat: float, lon: float, client: httpx.AsyncClient
) -> tuple[Optional[HourlyRow], list[HourlyRow], list[DailyAggregateRow], list[str]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weather_code,precipitation,relative_humidity_2m,apparent_temperature,dew_point_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,surface_pressure",
        "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,weather_code,surface_pressure,visibility,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,surface_pressure_mean",
        "timezone": "UTC",
        "wind_speed_unit": "ms"
    }
    url = "https://api.open-meteo.com/v1/forecast"

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=60.0)
        r.raise_for_status()
        return r.json()

    data = await with_exponential_backoff(_call, name="openmeteo_forecast")
    codes: list[str] = []

    hourly_raw = data.get("hourly", {})
    hourly: list[HourlyRow] = []
    if "time" in hourly_raw:
        for i in range(len(hourly_raw["time"])):
            hourly.append(_hourly_row(hourly_raw, i))

    if len(hourly) < 168: # Open-Meteo typically returns 7 days of hourly data (168h)
        codes.append(DegradationCode.OPEN_METEO_HOURLY_TRUNCATED.value) # Using same degradation code for backwards compat or we should rename it

    daily_ext: list[DailyAggregateRow] = []
    daily_raw = data.get("daily", {})
    if "time" in daily_raw:
        for i in range(len(daily_raw["time"])):
            daily_ext.append(_daily_row(daily_raw, i))

    if daily_ext:
        codes.append(DegradationCode.DAILY_EXTENDED_FROM_OPEN_METEO.value)

    current_raw = data.get("current")
    current_row: Optional[HourlyRow] = None
    if current_raw:
        try:
            current_row = _current_row(current_raw)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("openmeteo current parse failed: %s", e)
            codes.append(DegradationCode.CURRENT_CONDITIONS_MISSING.value)
    else:
        codes.append(DegradationCode.CURRENT_CONDITIONS_MISSING.value)

    return current_row, hourly, daily_ext, codes
