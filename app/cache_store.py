from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from app.config import ensure_cache_dir

logger = logging.getLogger(__name__)


def _path(kind: str, key: str) -> Path:
    base = ensure_cache_dir() / kind
    base.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key[:200])
    return base / f"{safe}.json"


def cache_get_json(kind: str, key: str) -> Optional[dict[str, Any]]:
    p = _path(kind, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("cache read failed %s: %s", p, e)
        return None


def cache_set_json(kind: str, key: str, data: dict[str, Any]) -> None:
    p = _path(kind, key)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
