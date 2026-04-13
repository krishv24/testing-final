from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

import httpx
import meteostat as ms
import pandas as pd

from app.assistant.retry import with_exponential_backoff
from app.config import get_settings
from app.degradation import DegradationCode
from app.schemas import ClimatologyBlock, MonthlyClimateRow

logger = logging.getLogger(__name__)


def _meteostat_monthly_normals_sync(
    lat: float, lon: float, elevation_m: Optional[float]
) -> tuple[Optional[str], list[MonthlyClimateRow]]:
    """
    Station search + monthly climate normals via the Meteostat Python package
    (bulk data; no API key). Uses WMO-style period from settings.
    """
    settings = get_settings()
    y0 = settings.meteostat_normals_start_year
    y1 = settings.meteostat_normals_end_year

    elev = int(round(elevation_m)) if elevation_m is not None else None
    point = ms.Point(lat, lon, elev)
    stations_df = ms.stations.nearby(point, limit=1)
    if stations_df is None or stations_df.empty:
        return None, []

    station_id = str(stations_df.index[0])
    ts = ms.normals(ms.Station(id=station_id), y0, y1)
    df = ts.fetch()
    if df is None or df.empty:
        return station_id, []

    rows: list[MonthlyClimateRow] = []
    for idx in df.index:
        try:
            mo = int(idx)
        except (TypeError, ValueError):
            continue
        if not 1 <= mo <= 12:
            continue
        row = df.loc[idx]
        tmin = float(row["tmin"]) if pd.notna(row.get("tmin")) else None
        tmax = float(row["tmax"]) if pd.notna(row.get("tmax")) else None
        prcp = row.get("prcp")
        ptot = float(prcp) if prcp is not None and pd.notna(prcp) else None
        rows.append(
            MonthlyClimateRow(month=mo, tmin_c=tmin, tmax_c=tmax, ptot_mm=ptot)
        )
    rows.sort(key=lambda r: r.month)
    return station_id, rows


async def _open_meteo_era5_monthly_normals(lat: float, lon: float, client: httpx.AsyncClient) -> list[MonthlyClimateRow]:
    """
    ERA5 reanalysis (nearest grid cell) via Open-Meteo Archive API.
    Aggregates 1991–2020 daily fields into monthly normals (WMO-style means).
    """
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1991-01-01",
        "end_date": "2020-12-31",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
    }

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=180.0)
        r.raise_for_status()
        return r.json()

    data = await with_exponential_backoff(_call, name="open_meteo_era5_archive")
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    tmin = daily.get("temperature_2m_min") or []
    tmax = daily.get("temperature_2m_max") or []
    prcp = daily.get("precipitation_sum") or []

    tmin_by_m: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    tmax_by_m: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    p_month_totals: dict[tuple[int, int], float] = {}

    for i, tstr in enumerate(times):
        try:
            dt = datetime.strptime(str(tstr)[:10], "%Y-%m-%d")
        except ValueError:
            continue
        mo = dt.month
        y = dt.year
        if i < len(tmin) and tmin[i] is not None:
            tmin_by_m[mo].append(float(tmin[i]))
        if i < len(tmax) and tmax[i] is not None:
            tmax_by_m[mo].append(float(tmax[i]))
        p = float(prcp[i]) if i < len(prcp) and prcp[i] is not None else 0.0
        key = (y, mo)
        p_month_totals[key] = p_month_totals.get(key, 0.0) + p

    rows: list[MonthlyClimateRow] = []
    for mo in range(1, 13):
        tmi = sum(tmin_by_m[mo]) / len(tmin_by_m[mo]) if tmin_by_m[mo] else None
        tma = sum(tmax_by_m[mo]) / len(tmax_by_m[mo]) if tmax_by_m[mo] else None
        month_precip_totals = [v for (y, m), v in p_month_totals.items() if m == mo]
        ppt = sum(month_precip_totals) / len(month_precip_totals) if month_precip_totals else None
        rows.append(MonthlyClimateRow(month=mo, tmin_c=tmi, tmax_c=tma, ptot_mm=ppt))
    return rows


async def fetch_climatology(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
    elevation_m: Optional[float] = None,
) -> tuple[ClimatologyBlock, list[str]]:
    codes: list[str] = []
    try:
        station_id, monthly = await asyncio.to_thread(
            _meteostat_monthly_normals_sync, lat, lon, elevation_m
        )
        if monthly:
            return (
                ClimatologyBlock(source="meteostat", station_id=station_id, monthly=monthly),
                codes,
            )
        if station_id and not monthly:
            logger.info("meteostat station %s returned empty normals; falling back", station_id)
    except Exception as e:
        logger.warning("meteostat library failed: %s", e)
        codes.append(DegradationCode.METEOSTAT_UNAVAILABLE.value)

    try:
        monthly = await _open_meteo_era5_monthly_normals(lat, lon, client)
        if monthly:
            codes.append(DegradationCode.ERA5_OPENMETEO_USED.value)
            return ClimatologyBlock(source="era5_open_meteo", station_id=None, monthly=monthly), codes
    except Exception as e:
        logger.warning("open-meteo ERA5 fallback failed: %s", e)

    codes.append(DegradationCode.CLIMATOLOGY_UNAVAILABLE.value)
    return ClimatologyBlock(source="unavailable", monthly=[]), codes
