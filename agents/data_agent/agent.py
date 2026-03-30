"""
DataAgent — the single entry point for all FPL data queries.

It composes FPLClient (HTTP + cache) and the Pydantic models to return
strongly-typed objects to callers.  Future agents (TransferAgent,
LineupAgent, etc.) will depend only on this module, never on the HTTP
client directly.
"""

from __future__ import annotations

from typing import Optional

from .client import FPLClient
from .models import BootstrapData, Fixture, Player, PlayerDetail, PlayerGameweekHistory, Team


class DataAgent:
    """
    Async facade over the FPL API.

    Usage::

        async with DataAgent() as agent:
            bootstrap = await agent.bootstrap()
            salah = await agent.get_player_detail(308)
    """

    def __init__(self, timeout: float = 15.0) -> None:
        self._client = FPLClient(timeout=timeout)

    # ------------------------------------------------------------------
    # Context manager — delegates lifecycle to the HTTP client
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DataAgent":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.__aexit__(*args)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def bootstrap(self) -> BootstrapData:
        """
        Fetch and parse the bootstrap-static endpoint.

        Returns teams, all players, and the current gameweek number.
        This is the primary data source — most callers should start here.
        """
        raw = await self._client.get_bootstrap()
        teams = [Team.model_validate(t) for t in raw["teams"]]
        players = [Player.model_validate(p) for p in raw["elements"]]
        current_gw = _current_gameweek(raw.get("events", []))
        return BootstrapData(teams=teams, players=players, current_gameweek=current_gw)

    async def get_players(self) -> list[Player]:
        """Return all players with current stats and prices."""
        data = await self.bootstrap()
        return data.players

    async def get_teams(self) -> list[Team]:
        """Return all 20 Premier League teams."""
        data = await self.bootstrap()
        return data.teams

    async def get_fixtures(self, gameweek: Optional[int] = None) -> list[Fixture]:
        """
        Return fixtures for the full season or a specific gameweek.

        Args:
            gameweek: If given, restrict to that GW only.
        """
        raw_fixtures = await self._client.get_fixtures(gameweek)
        return [Fixture.model_validate(f) for f in raw_fixtures]

    async def get_player_detail(self, player_id: int) -> PlayerDetail:
        """
        Return a player's gameweek-by-gameweek history and upcoming fixtures.

        Args:
            player_id: The FPL element id (visible in bootstrap players).
        """
        raw = await self._client.get_element_summary(player_id)
        history = [PlayerGameweekHistory.model_validate(h) for h in raw.get("history", [])]
        upcoming = [Fixture.model_validate(f) for f in raw.get("fixtures", [])]
        return PlayerDetail(player_id=player_id, history=history, fixtures=upcoming)

    async def find_players(
        self,
        *,
        position: Optional[str] = None,
        team_id: Optional[int] = None,
        max_price: Optional[float] = None,
        min_total_points: int = 0,
    ) -> list[Player]:
        """
        Filter the player list by position, team, price cap, or minimum points.

        Args:
            position: "GKP", "DEF", "MID", or "FWD" (case-insensitive).
            team_id:  Filter to one team's players.
            max_price: Upper bound in £m (e.g. 7.5).
            min_total_points: Only include players above this points threshold.
        """
        players = await self.get_players()

        if position:
            pos_upper = position.upper()
            players = [p for p in players if p.position == pos_upper]

        if team_id is not None:
            players = [p for p in players if p.team == team_id]

        if max_price is not None:
            players = [p for p in players if p.price <= max_price]

        if min_total_points:
            players = [p for p in players if p.total_points >= min_total_points]

        return players


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _current_gameweek(events: list[dict]) -> Optional[int]:
    """
    Derive the current (or most recently finished) gameweek id from the
    events list returned by bootstrap-static.
    """
    for event in events:
        if event.get("is_current"):
            return event["id"]
    # Fall back to the most recent finished GW.
    finished = [e for e in events if e.get("finished")]
    if finished:
        return finished[-1]["id"]
    return None
