"""
Pydantic models for FPL API responses.

All monetary values (prices) are stored in tenths of millions as returned
by the API (e.g. 65 = £6.5m) and exposed as a float property for convenience.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from typing import Optional

from pydantic import BaseModel, computed_field, Field


class Position(IntEnum):
    GKP = 1
    DEF = 2
    MID = 3
    FWD = 4


class Team(BaseModel):
    id: int
    name: str
    short_name: str
    strength: int
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int


class Player(BaseModel):
    id: int
    first_name: str
    second_name: str
    web_name: str
    team: int                          # team id
    element_type: Position             # 1=GKP 2=DEF 3=MID 4=FWD
    now_cost: int                      # raw: tenths of £m (65 = £6.5m)
    total_points: int
    form: float = Field(alias="form", default=0.0)
    points_per_game: float = Field(alias="points_per_game", default=0.0)
    selected_by_percent: float = Field(alias="selected_by_percent", default=0.0)
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0                       # bonus point system score
    influence: float = Field(default=0.0)
    creativity: float = Field(default=0.0)
    threat: float = Field(default=0.0)
    ict_index: float = Field(default=0.0)
    transfers_in_event: int = 0
    transfers_out_event: int = 0
    status: str = "a"                  # a=available, d=doubtful, i=injured, s=suspended, u=unavailable

    model_config = {"populate_by_name": True}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price(self) -> float:
        """Price in £m (e.g. 6.5)."""
        return self.now_cost / 10

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.second_name}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def position(self) -> str:
        return self.element_type.name


class Fixture(BaseModel):
    id: int
    event: Optional[int] = None        # gameweek number; None = not yet scheduled
    kickoff_time: Optional[datetime] = None
    team_h: int                        # home team id
    team_a: int                        # away team id
    team_h_difficulty: int             # FDR 1–5
    team_a_difficulty: int
    finished: bool = False
    finished_provisional: bool = False
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None
    minutes: int = 0                   # minutes played (live)
    started: Optional[bool] = None


class PlayerGameweekHistory(BaseModel):
    """Single gameweek row from element-summary endpoint."""
    fixture: int
    opponent_team: int
    total_points: int
    was_home: bool
    kickoff_time: Optional[datetime] = None
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None
    round: int                         # gameweek
    minutes: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    saves: int = 0
    bonus: int = 0
    bps: int = 0
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    ict_index: float = 0.0
    value: int = 0                     # price at time of fixture (tenths)
    selected: int = 0
    transfers_in: int = 0
    transfers_out: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_at_fixture(self) -> float:
        return self.value / 10


class PlayerDetail(BaseModel):
    """Full player detail including gameweek history."""
    player_id: int
    history: list[PlayerGameweekHistory]
    fixtures: list[Fixture]            # upcoming fixtures for this player


class BootstrapData(BaseModel):
    """Parsed bootstrap-static response."""
    teams: list[Team]
    players: list[Player]
    current_gameweek: Optional[int]
