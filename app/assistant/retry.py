import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_RETRIES = 5
BASE_DELAY = 0.5
MAX_DELAY = 16.0


async def with_exponential_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = DEFAULT_RETRIES,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    name: str = "request",
) -> T:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return await fn()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            if isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code not in (429, 500, 502, 503, 504):
                    raise
            if attempt == retries - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            logger.warning("%s failed attempt %s/%s: %s; retry in %.2fs", name, attempt + 1, retries, e, delay)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
