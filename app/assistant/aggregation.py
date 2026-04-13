from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np

from app.schemas import DailyAggregateRow, HourlyRow, SixHourAggregateRow


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def circular_mean_deg(degrees: Iterable[float]) -> float:
    arr = np.array([d for d in degrees if d is not None], dtype=float)
    if arr.size == 0:
        return float("nan")
    rad = np.deg2rad(arr)
    y = np.mean(np.sin(rad))
    x = np.mean(np.cos(rad))
    ang = float(np.degrees(np.arctan2(y, x)) % 360.0)
    return ang


def aggregate_six_hour(hourly: list[HourlyRow]) -> list[SixHourAggregateRow]:
    buckets: dict[tuple[str, int], list[HourlyRow]] = defaultdict(list)
    for h in hourly:
        t = _parse_ts(h.timestamp_utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        day = t.date().isoformat()
        slot = t.hour // 6
        buckets[(day, slot)].append(h)

    rows: list[SixHourAggregateRow] = []
    for (day, slot), hs in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        hs_sorted = sorted(hs, key=lambda x: x.timestamp_utc)
        t0 = _parse_ts(hs_sorted[0].timestamp_utc)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        start = datetime(t0.year, t0.month, t0.day, slot * 6, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(hours=6)
        temps = [x.t_c for x in hs_sorted]
        rhs = [x.rh_percent for x in hs_sorted]
        ws = [x.wind_speed_ms for x in hs_sorted]
        wdeg = [x.wind_direction_deg for x in hs_sorted]
        pr = [x.precipitation_mm for x in hs_sorted]
        td = [x.td_c for x in hs_sorted]
        vis = [x.visibility_m for x in hs_sorted if x.visibility_m is not None]
        prs = [x.pressure_hpa for x in hs_sorted if x.pressure_hpa is not None]
        wd_mean = circular_mean_deg(wdeg)
        rows.append(
            SixHourAggregateRow(
                window_start_utc=start.isoformat(),
                window_end_utc=end.isoformat(),
                t_mean_c=float(np.mean(temps)),
                t_min_c=float(np.min(temps)),
                t_max_c=float(np.max(temps)),
                rh_mean_percent=float(np.mean(rhs)),
                wind_speed_mean_ms=float(np.mean(ws)),
                wind_direction_mean_deg=wd_mean if not np.isnan(wd_mean) else None,
                precipitation_sum_mm=float(np.sum(pr)),
                td_mean_c=float(np.mean(td)),
                visibility_mean_m=float(np.mean(vis)) if vis else None,
                pressure_mean_hpa=float(np.mean(prs)) if prs else None,
            )
        )
    return rows


def aggregate_daily_from_hourly(hourly: list[HourlyRow]) -> list[DailyAggregateRow]:
    by_day: dict[str, list[HourlyRow]] = defaultdict(list)
    for h in hourly:
        t = _parse_ts(h.timestamp_utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        by_day[t.date().isoformat()].append(h)

    rows: list[DailyAggregateRow] = []
    for day in sorted(by_day.keys()):
        hs = sorted(by_day[day], key=lambda x: x.timestamp_utc)
        temps = [x.t_c for x in hs]
        rhs = [x.rh_percent for x in hs]
        ws = [x.wind_speed_ms for x in hs]
        pr = [x.precipitation_mm for x in hs]
        td = [x.td_c for x in hs]
        vis = [x.visibility_m for x in hs if x.visibility_m is not None]
        prs = [x.pressure_hpa for x in hs if x.pressure_hpa is not None]
        rows.append(
            DailyAggregateRow(
                date_utc=day,
                aggregate_source="hourly_derived",
                t_mean_c=float(np.mean(temps)),
                t_min_c=float(np.min(temps)),
                t_max_c=float(np.max(temps)),
                rh_mean_percent=float(np.mean(rhs)),
                wind_speed_mean_ms=float(np.mean(ws)),
                precipitation_sum_mm=float(np.sum(pr)),
                td_mean_c=float(np.mean(td)),
                visibility_mean_m=float(np.mean(vis)) if vis else None,
                pressure_mean_hpa=float(np.mean(prs)) if prs else None,
            )
        )
    return rows


def merge_daily_tables(
    hourly_derived: list[DailyAggregateRow],
    open_meteo_daily: list[DailyAggregateRow],
) -> list[DailyAggregateRow]:
    """Prefer hourly_derived for overlapping dates; append OW daily for extra days."""
    by_date = {d.date_utc: d for d in hourly_derived}
    out = list(hourly_derived)
    for d in sorted(open_meteo_daily, key=lambda x: x.date_utc):
        if d.date_utc not in by_date:
            out.append(d)
    out.sort(key=lambda x: x.date_utc)
    return out
