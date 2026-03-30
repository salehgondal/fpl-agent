"""
Thin async HTTP client for the FPL public API.

Responsibilities:
  - Own all URL construction and request headers.
  - Apply a simple TTL in-memory cache so callers that invoke multiple
    methods in one agent run don't hammer the same endpoint repeatedly.
  - Return raw dicts/lists; parsing into models is the agent's job.

The cache is intentionally primitive (dict + timestamp) to keep this module
dependency-free beyond httpx.  A production version could swap in Redis or
diskcache without touching any other layer.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

_BASE = "https://fantasy.premierleague.com/api"

# Endpoints that rarely change mid-GW get a longer TTL.
_TTL: dict[str, int] = {
    "bootstrap-static": 300,   # 5 minutes
    "fixtures":         120,   # 2 minutes
    "element-summary":  60,    # 1 minute
}
_DEFAULT_TTL = 60


class FPLClient:
    """
    Async HTTP client for the FPL REST API.

    Designed to be used as an async context manager so the underlying
    httpx.AsyncClient is properly closed:

        async with FPLClient() as client:
            data = await client.get_bootstrap()
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout
        self._cache: dict[str, tuple[Any, float]] = {}
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "FPLClient":
        self._http = httpx.AsyncClient(
            base_url=_BASE,
            timeout=self._timeout,
            headers={
                # FPL blocks default httpx UA; mimic a browser.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_bootstrap(self) -> dict[str, Any]:
        """Return the full bootstrap-static payload."""
        return await self._get("/bootstrap-static/", ttl_key="bootstrap-static")

    async def get_fixtures(self, gameweek: int | None = None) -> list[dict[str, Any]]:
        """
        Return fixtures.  Pass gameweek to restrict to a single GW,
        or omit to get the full season schedule.
        """
        path = "/fixtures/" if gameweek is None else f"/fixtures/?event={gameweek}"
        return await self._get(path, ttl_key="fixtures")  # type: ignore[return-value]

    async def get_element_summary(self, player_id: int) -> dict[str, Any]:
        """Return history + upcoming fixtures for a single player."""
        return await self._get(
            f"/element-summary/{player_id}/",
            ttl_key="element-summary",
            cache_key=f"element-summary-{player_id}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(
        self,
        path: str,
        *,
        ttl_key: str = "",
        cache_key: str = "",
    ) -> Any:
        key = cache_key or path
        ttl = _TTL.get(ttl_key, _DEFAULT_TTL)

        cached_value, cached_at = self._cache.get(key, (None, 0.0))
        if cached_value is not None and (time.monotonic() - cached_at) < ttl:
            return cached_value

        if self._http is None:
            raise RuntimeError(
                "FPLClient must be used as an async context manager. "
                "Use 'async with FPLClient() as client: ...'"
            )

        response = await self._http.get(path)
        response.raise_for_status()
        data = response.json()
        self._cache[key] = (data, time.monotonic())
        return data
