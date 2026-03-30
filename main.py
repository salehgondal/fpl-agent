"""
FPL Agent — interactive data explorer.

Usage:
    python main.py                  # show overview + top players
    python main.py fixtures         # upcoming fixtures this gameweek
    python main.py player <id>      # detail for a specific player id
    python main.py find <position>  # e.g. find MID, find FWD
"""

import asyncio
import sys
from agents.data_agent import DataAgent


async def overview(agent: DataAgent) -> None:
    bootstrap = await agent.bootstrap()
    print(f"\n=== Gameweek {bootstrap.current_gameweek} ===")
    print(f"Teams: {len(bootstrap.teams)}  |  Players: {len(bootstrap.players)}\n")

    positions = ["GKP", "DEF", "MID", "FWD"]
    for pos in positions:
        top = sorted(
            [p for p in bootstrap.players if p.position == pos],
            key=lambda p: p.total_points,
            reverse=True,
        )[:5]
        print(f"Top 5 {pos}s:")
        for p in top:
            print(
                f"  {p.web_name:<22} £{p.price:.1f}m  "
                f"{p.total_points:>3}pts  form={p.form:.1f}  "
                f"sel={p.selected_by_percent:.1f}%"
            )
        print()


async def fixtures(agent: DataAgent) -> None:
    bootstrap = await agent.bootstrap()
    gw = bootstrap.current_gameweek
    team_map = {t.id: t.short_name for t in bootstrap.teams}

    all_fixtures = await agent.get_fixtures(gameweek=gw)
    print(f"\n=== GW{gw} Fixtures ===")
    for f in sorted(all_fixtures, key=lambda x: x.kickoff_time or ""):
        home = team_map.get(f.team_h, str(f.team_h))
        away = team_map.get(f.team_a, str(f.team_a))
        ko = f.kickoff_time.strftime("%a %d %b %H:%M") if f.kickoff_time else "TBC"
        if f.finished:
            print(f"  {home} {f.team_h_score}–{f.team_a_score} {away}  [{ko}]")
        else:
            print(f"  {home} vs {away}  (FDR {f.team_h_difficulty} / {f.team_a_difficulty})  [{ko}]")


async def player_detail(agent: DataAgent, player_id: int) -> None:
    bootstrap = await agent.bootstrap()
    team_map = {t.id: t.short_name for t in bootstrap.teams}
    player_map = {p.id: p for p in bootstrap.players}

    p = player_map.get(player_id)
    if not p:
        print(f"No player found with id {player_id}")
        return

    detail = await agent.get_player_detail(player_id)

    print(f"\n=== {p.full_name} ({p.web_name}) ===")
    print(f"Team: {team_map.get(p.team, '?')}  |  Position: {p.position}  |  Price: £{p.price:.1f}m")
    print(f"Total pts: {p.total_points}  |  Form: {p.form}  |  ICT: {p.ict_index}")
    print(f"Goals: {p.goals_scored}  Assists: {p.assists}  CS: {p.clean_sheets}  Bonus: {p.bonus}")
    print(f"Selected by: {p.selected_by_percent}%  |  Status: {p.status}\n")

    if detail.history:
        print("Last 5 gameweeks:")
        for h in detail.history[-5:]:
            opp = team_map.get(h.opponent_team, str(h.opponent_team))
            venue = "H" if h.was_home else "A"
            print(
                f"  GW{h.round:<3} vs {opp:<4} ({venue})  "
                f"{h.minutes}min  {h.total_points}pts  "
                f"G:{h.goals_scored} A:{h.assists} CS:{h.clean_sheets} B:{h.bonus}"
            )

    if detail.fixtures:
        print("\nNext 5 fixtures:")
        for f in detail.fixtures[:5]:
            opp_id = f.team_a if p.team == f.team_h else f.team_h
            fdr   = f.team_h_difficulty if p.team == f.team_h else f.team_a_difficulty
            venue = "H" if p.team == f.team_h else "A"
            opp   = team_map.get(opp_id, str(opp_id))
            ko    = f.kickoff_time.strftime("%a %d %b") if f.kickoff_time else "TBC"
            print(f"  GW{f.event}  vs {opp:<4} ({venue})  FDR={fdr}  [{ko}]")


async def find_players(agent: DataAgent, position: str) -> None:
    players = await agent.find_players(position=position, min_total_points=1)
    players.sort(key=lambda p: p.total_points, reverse=True)

    print(f"\n=== {position.upper()} players ranked by total points ===")
    print(f"  {'Name':<22} {'£':>5}  {'Pts':>4}  {'Form':>5}  {'PPG':>5}  {'Sel%':>6}")
    print("  " + "-" * 58)
    for p in players[:20]:
        print(
            f"  {p.web_name:<22} £{p.price:.1f}  {p.total_points:>4}  "
            f"{p.form:>5.1f}  {p.points_per_game:>5.1f}  {p.selected_by_percent:>5.1f}%"
        )


async def main() -> None:
    args = sys.argv[1:]

    async with DataAgent() as agent:
        if not args:
            await overview(agent)
        elif args[0] == "fixtures":
            await fixtures(agent)
        elif args[0] == "player" and len(args) == 2:
            await player_detail(agent, int(args[1]))
        elif args[0] == "find" and len(args) == 2:
            await find_players(agent, args[1])
        else:
            print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
