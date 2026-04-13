from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.assistant.retry import with_exponential_backoff
from app.config import get_settings
from app.schemas import LocationMeta

logger = logging.getLogger(__name__)


async def _reverse_geocode(lat: float, lon: float, client: httpx.AsyncClient) -> Optional[dict[str, Any]]:
    settings = get_settings()
    params = {"lat": lat, "lng": lon, "username": settings.geonames_username, "style": "FULL", "maxRows": 1}
    url = f"{settings.geonames_base}/findNearbyPlaceNameJSON"

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=60.0)
        r.raise_for_status()
        return r.json()

    data = await with_exponential_backoff(_call, name="geonames_reverse")
    geonames = data.get("geonames") or []
    return geonames[0] if geonames else None


async def fetch_geonames_search(q: str, client: httpx.AsyncClient) -> dict[str, Any]:
    settings = get_settings()
    params = {
        "q": q,
        "maxRows": 1,
        "username": settings.geonames_username,
        "style": "FULL",
    }
    url = f"{settings.geonames_base}/searchJSON"

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=60.0)
        r.raise_for_status()
        return r.json()

    return await with_exponential_backoff(_call, name="geonames_search")


async def fetch_geonames_elevation(lat: float, lon: float, client: httpx.AsyncClient) -> Optional[float]:
    settings = get_settings()
    params = {"lat": lat, "lng": lon, "username": settings.geonames_username}
    url = f"{settings.geonames_base}/astergdemJSON"

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=60.0)
        r.raise_for_status()
        return r.json()

    try:
        data = await with_exponential_backoff(_call, name="geonames_srtm")
        if "astergdem" in data and data["astergdem"] is not None:
            return float(data["astergdem"])
    except Exception as e:
        logger.info("elevation lookup failed: %s", e)
    return None


async def fetch_wikipedia_summary(title_hint: str, client: httpx.AsyncClient) -> Optional[str]:
    """Short extract via MediaWiki API (English)."""
    if not title_hint:
        return None
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "titles": title_hint,
        "redirects": "1",
    }

    async def _call() -> dict[str, Any]:
        r = await client.get(url, params=params, timeout=30.0, headers={"User-Agent": "AIMeteorologist/1.0"})
        r.raise_for_status()
        return r.json()

    try:
        data = await with_exponential_backoff(_call, name="wikipedia")
        pages = data.get("query", {}).get("pages", {})
        for _pid, page in pages.items():
            ext = page.get("extract")
            if ext:
                return ext[:1200]
    except Exception as e:
        logger.info("wikipedia failed: %s", e)
    return None


async def resolve_location(
    *,
    query: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    client: httpx.AsyncClient,
) -> LocationMeta:
    if lat is not None and lon is not None:
        el = await fetch_geonames_elevation(lat, lon, client)
        city = query or f"{lat:.4f},{lon:.4f}"
        admin = ""
        country = ""
        try:
            rev = await _reverse_geocode(lat, lon, client)
            if rev:
                city = rev.get("name") or city
                admin = rev.get("adminName1") or ""
                country = rev.get("countryName") or ""
        except Exception:
            pass
        return LocationMeta(
            city=city,
            administrative_region=admin,
            country=country,
            latitude=lat,
            longitude=lon,
            elevation_m=el,
        )

    if not query:
        raise ValueError("Either query or lat/lon is required")

    data = await fetch_geonames_search(query, client)
    geonames = data.get("geonames") or []
    if not geonames:
        raise ValueError(f"No GeoNames result for query: {query}")
    g = geonames[0]
    lat_f = float(g["lat"])
    lon_f = float(g["lng"])
    el = g.get("srtm3") or g.get("astergdem")
    if el is None:
        el = await fetch_geonames_elevation(lat_f, lon_f, client)
    elif el is not None:
        el = float(el)

    city = g.get("name") or query
    admin = g.get("adminName1") or ""
    country = g.get("countryName") or ""
    wiki = await fetch_wikipedia_summary(city, client)

    return LocationMeta(
        city=city,
        administrative_region=admin,
        country=country,
        latitude=lat_f,
        longitude=lon_f,
        elevation_m=el,
        geoname_id=int(g.get("geonameId", 0)) or None,
        wikipedia_summary=wiki,
    )
