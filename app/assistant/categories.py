"""Rule-based compact weather category from WMO weather codes (Open-Meteo)."""

from __future__ import annotations


def category_from_wmo_code(code: int, precipitation_mm: float = 0.0) -> str:
    c = int(code)
    # Clear / clouds
    if c == 0:
        return "clear"
    if c == 1:
        return "few_clouds"
    if c == 2:
        return "scattered_clouds"
    if c == 3:
        return "overcast"
    
    # Fog
    if c in (45, 48):
        return "fog"
        
    # Drizzle
    if c in (51, 53, 55, 56, 57):
        return "drizzle"

    # Rain
    if c in (61, 63, 65, 66, 67, 80, 81, 82):
        if c in (65, 67, 82) or precipitation_mm >= 3.0:
            return "heavy_rain"
        return "rain"

    # Snow
    if c in (71, 73, 75, 77, 85, 86):
        return "snow"

    # Thunderstorm
    if c in (95, 96, 99):
        return "thunderstorm"

    return "unknown"

def label_from_wmo_code(code: int) -> str:
    labels = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Drizzle: Light",
        53: "Drizzle: Moderate",
        55: "Drizzle: Dense",
        56: "Freezing Drizzle: Light",
        57: "Freezing Drizzle: Dense",
        61: "Rain: Slight",
        63: "Rain: Moderate",
        65: "Rain: Heavy",
        66: "Freezing Rain: Light",
        67: "Freezing Rain: Heavy",
        71: "Snow fall: Slight",
        73: "Snow fall: Moderate",
        75: "Snow fall: Heavy",
        77: "Snow grains",
        80: "Rain showers: Slight",
        81: "Rain showers: Moderate",
        82: "Rain showers: Violent",
        85: "Snow showers: Slight",
        86: "Snow showers: Heavy",
        95: "Thunderstorm: Slight or moderate",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }
    return labels.get(int(code), "Unknown")
