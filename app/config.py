from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "app/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # openweather_api_key: str = "" (removed in favor of Open-Meteo)
    geonames_username: str = "demo"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    cds_url: str = ""
    cds_key: str = ""
    cache_dir: Path = Path(".cache")

    geonames_base: str = "http://api.geonames.org"

    # Climate normals period for Meteostat Python library (WMO-style; matches ERA5 fallback window)
    meteostat_normals_start_year: int = 1991
    meteostat_normals_end_year: int = 2020


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_cache_dir() -> Path:
    s = get_settings()
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s.cache_dir
