import pandas as pd
from typing import Any

def normalize_to_utc(series_or_val: Any) -> Any:
    """Ensures both naive and aware datetimes are converted to timezone-aware UTC."""
    if isinstance(series_or_val, pd.Series):
        dt_series = pd.to_datetime(series_or_val)
        if dt_series.dt.tz is None:
            return dt_series.dt.tz_localize("UTC")
        else:
            return dt_series.dt.tz_convert("UTC")
    else:
        ts = pd.to_datetime(series_or_val)
        if ts.tz is None:
            return ts.tz_localize("UTC")
        else:
            return ts.tz_convert("UTC")
