"""NOAA NWS Area Forecast Discussion (US only)."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import httpx

from app.assistant.retry import with_exponential_backoff
from app.degradation import DegradationCode
from app.schemas import LocationMeta

logger = logging.getLogger(__name__)

NWS_USER_AGENT = "AIMeteorologist/1.0 (research; contact@localhost)"


async def enrich_us_nws_grid(lat: float, lon: float, client: httpx.AsyncClient) -> dict[str, Any]:
    url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}

    async def _call() -> dict[str, Any]:
        r = await client.get(url, headers=headers, timeout=30.0)
        r.raise_for_status()
        return r.json()

    data = await with_exponential_backoff(_call, name="nws_points")
    props = data.get("properties") or {}
    grid_id = props.get("gridId")
    gx = props.get("gridX")
    gy = props.get("gridY")
    cwa = props.get("cwa")
    return {"grid_id": grid_id, "grid_x": gx, "grid_y": gy, "cwa": cwa}


async def fetch_latest_afd_text(office: str, client: httpx.AsyncClient) -> Optional[str]:
    """Fetch latest AFD product text for a weather forecast office (e.g. LOT, MTR)."""
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    list_url = f"https://api.weather.gov/products/types/AFD/locations/{office}"

    async def _list() -> dict[str, Any]:
        r = await client.get(list_url, headers=headers, timeout=30.0)
        r.raise_for_status()
        return r.json()

    try:
        meta = await with_exponential_backoff(_list, name="nws_afd_list")
        graph = meta.get("@graph") or meta.get("locations") or []
        if isinstance(meta, list):
            graph = meta
        if not graph:
            return None
        latest = graph[0]
        pid = latest.get("id") or latest.get("@id", "").split("/")[-1]
        if not pid:
            return None
        product_url = f"https://api.weather.gov/products/{pid}"

        async def _prod() -> dict[str, Any]:
            r = await client.get(product_url, headers=headers, timeout=30.0)
            r.raise_for_status()
            return r.json()

        prod = await with_exponential_backoff(_prod, name="nws_afd_product")
        text = prod.get("productText") or ""
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()[:8000] if text else None
    except Exception as e:
        logger.info("NWS AFD fetch failed: %s", e)
        return None


async def maybe_attach_afd(loc: LocationMeta, client: httpx.AsyncClient) -> tuple[LocationMeta, Optional[str], list[str]]:
    codes: list[str] = []
    # US approximate: lat 24-50, lon -125 to -66
    if not (-126 < loc.longitude < -65 and 23 < loc.latitude < 51):
        codes.append(DegradationCode.NWS_AFD_SKIPPED_NON_US.value)
        return loc, None, codes
    try:
        grid = await enrich_us_nws_grid(loc.latitude, loc.longitude, client)
        cwa = grid.get("cwa")
        if cwa:
            loc = loc.model_copy(
                update={
                    "nws_forecast_office": str(cwa),
                    "nws_grid_x": grid.get("grid_x"),
                    "nws_grid_y": grid.get("grid_y"),
                }
            )
        text = await fetch_latest_afd_text(str(cwa), client) if cwa else None
        if not text:
            codes.append(DegradationCode.NWS_AFD_UNAVAILABLE.value)
        return loc, text, codes
    except Exception as e:
        logger.info("NWS enrichment failed: %s", e)
        codes.append(DegradationCode.NWS_AFD_UNAVAILABLE.value)
        return loc, None, codes
