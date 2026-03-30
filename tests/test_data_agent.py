"""
Unit tests for DataAgent.

All HTTP calls are intercepted by respx so no real network traffic is made.
Fixtures (pytest) provide minimal but valid FPL API response shapes so the
Pydantic models can parse them end-to-end.
"""

from __future__ import annotations

import pytest
import respx
import httpx

from agents.data_agent import DataAgent
from agents.data_agent.models import Position


# ---------------------------------------------------------------------------
# Shared fake API data
# ---------------------------------------------------------------------------

FAKE_BOOTSTRAP: dict = {
    "teams": [
        {
            "id": 1,
            "name": "Arsenal",
            "short_name": "ARS",
            "strength": 4,
            "strength_overall_home": 1280,
            "strength_overall_away": 1270,
            "strength_attack_home": 1290,
            "strength_attack_away": 1280,
            "strength_defence_home": 1260,
            "strength_defence_away": 1250,
        }
    ],
    "elements": [
        {
            "id": 1,
            "first_name": "Bukayo",
            "second_name": "Saka",
            "web_name": "Saka",
            "team": 1,
            "element_type": 3,
            "now_cost": 100,
            "total_points": 180,
            "form": "7.2",
            "points_per_game": "7.0",
            "selected_by_percent": "35.5",
            "minutes": 2500,
            "goals_scored": 14,
            "assists": 12,
            "clean_sheets": 8,
            "goals_conceded": 20,
            "yellow_cards": 2,
            "red_cards": 0,
            "saves": 0,
            "bonus": 30,
            "bps": 520,
            "influence": "800.0",
            "creativity": "900.0",
            "threat": "750.0",
            "ict_index": "245.0",
            "transfers_in_event": 50000,
            "transfers_out_event": 10000,
            "status": "a",
        }
    ],
    "events": [
        {"id": 28, "is_current": True, "finished": False},
        {"id": 27, "is_current": False, "finished": True},
    ],
}

FAKE_FIXTURES: list[dict] = [
    {
        "id": 1,
        "event": 28,
        "kickoff_time": "2025-03-15T15:00:00Z",
        "team_h": 1,
        "team_a": 2,
        "team_h_difficulty": 2,
        "team_a_difficulty": 4,
        "finished": False,
        "finished_provisional": False,
        "team_h_score": None,
        "team_a_score": None,
        "minutes": 0,
        "started": False,
    }
]

FAKE_ELEMENT_SUMMARY: dict = {
    "history": [
        {
            "fixture": 10,
            "opponent_team": 5,
            "total_points": 12,
            "was_home": True,
            "kickoff_time": "2025-01-10T20:00:00Z",
            "team_h_score": 3,
            "team_a_score": 0,
            "round": 21,
            "minutes": 90,
            "goals_scored": 1,
            "assists": 2,
            "clean_sheets": 1,
            "goals_conceded": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "saves": 0,
            "bonus": 3,
            "bps": 42,
            "influence": "70.0",
            "creativity": "80.0",
            "threat": "65.0",
            "ict_index": "21.5",
            "value": 100,
            "selected": 4500000,
            "transfers_in": 80000,
            "transfers_out": 10000,
        }
    ],
    "fixtures": FAKE_FIXTURES,
}

_BASE = "https://fantasy.premierleague.com/api"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@respx.mock
@pytest.mark.asyncio
async def test_bootstrap_returns_correct_types():
    respx.get(f"{_BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=FAKE_BOOTSTRAP)
    )
    async with DataAgent() as agent:
        data = await agent.bootstrap()

    assert len(data.teams) == 1
    assert data.teams[0].name == "Arsenal"
    assert len(data.players) == 1
    assert data.current_gameweek == 28


@respx.mock
@pytest.mark.asyncio
async def test_player_price_computed_field():
    respx.get(f"{_BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=FAKE_BOOTSTRAP)
    )
    async with DataAgent() as agent:
        players = await agent.get_players()

    saka = players[0]
    assert saka.price == 10.0
    assert saka.position == "MID"
    assert saka.full_name == "Bukayo Saka"


@respx.mock
@pytest.mark.asyncio
async def test_get_fixtures():
    respx.get(f"{_BASE}/fixtures/").mock(
        return_value=httpx.Response(200, json=FAKE_FIXTURES)
    )
    async with DataAgent() as agent:
        fixtures = await agent.get_fixtures()

    assert len(fixtures) == 1
    assert fixtures[0].team_h_difficulty == 2


@respx.mock
@pytest.mark.asyncio
async def test_get_player_detail():
    respx.get(f"{_BASE}/element-summary/1/").mock(
        return_value=httpx.Response(200, json=FAKE_ELEMENT_SUMMARY)
    )
    async with DataAgent() as agent:
        detail = await agent.get_player_detail(1)

    assert detail.player_id == 1
    assert len(detail.history) == 1
    assert detail.history[0].price_at_fixture == 10.0
    assert detail.history[0].goals_scored == 1


@respx.mock
@pytest.mark.asyncio
async def test_find_players_filter_by_position():
    respx.get(f"{_BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=FAKE_BOOTSTRAP)
    )
    async with DataAgent() as agent:
        mids = await agent.find_players(position="MID")
        fwds = await agent.find_players(position="FWD")

    assert len(mids) == 1
    assert len(fwds) == 0


@respx.mock
@pytest.mark.asyncio
async def test_find_players_filter_by_price():
    respx.get(f"{_BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=FAKE_BOOTSTRAP)
    )
    async with DataAgent() as agent:
        cheap = await agent.find_players(max_price=9.9)
        affordable = await agent.find_players(max_price=10.0)

    assert len(cheap) == 0
    assert len(affordable) == 1


@respx.mock
@pytest.mark.asyncio
async def test_cache_prevents_duplicate_requests():
    """Two calls to get_players() within one agent session hit the API once."""
    route = respx.get(f"{_BASE}/bootstrap-static/").mock(
        return_value=httpx.Response(200, json=FAKE_BOOTSTRAP)
    )
    async with DataAgent() as agent:
        await agent.get_players()
        await agent.get_players()

    assert route.call_count == 1
