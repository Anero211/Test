"""PandaScore API (бесплатный ключ, ~1000 запросов/мес): история и расписание матчей CS2.

Документация: https://developers.pandascore.co/reference/get_csgo_matches_past
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from ..storage import Match, parse_dt
from .http import Fetcher

log = logging.getLogger(__name__)
BASE = "https://api.pandascore.co/csgo"


def parse_match(obj: dict) -> Match | None:
    opps = [o.get("opponent") or {} for o in obj.get("opponents") or []]
    if len(opps) != 2 or not obj.get("winner_id"):
        return None
    scores = {r.get("team_id"): r.get("score", 0) for r in obj.get("results") or []}
    t1, t2 = opps
    dt = parse_dt(obj.get("begin_at") or obj.get("scheduled_at"))
    if dt is None or obj.get("forfeit"):
        return None
    league = (obj.get("league") or {}).get("name", "")
    serie = (obj.get("serie") or {}).get("full_name", "")
    return Match(
        match_id=f"ps:{obj['id']}", date=dt, team1=t1.get("name", ""), team2=t2.get("name", ""),
        score1=int(scores.get(t1.get("id"), 0)), score2=int(scores.get(t2.get("id"), 0)),
        best_of=int(obj.get("number_of_games") or 1), event=f"{league} {serie}".strip(), source="pandascore",
    )


def parse_upcoming(obj: dict) -> dict | None:
    opps = [o.get("opponent") or {} for o in obj.get("opponents") or []]
    if len(opps) != 2:
        return None
    return {"team1": opps[0].get("name", ""), "team2": opps[1].get("name", ""),
            "start": parse_dt(obj.get("begin_at") or obj.get("scheduled_at")),
            "best_of": obj.get("number_of_games"), "event": (obj.get("league") or {}).get("name", ""),
            "source": "pandascore"}


class PandaScoreClient:
    def __init__(self, token: str, fetcher: Fetcher | None = None):
        if not token:
            raise ValueError("Нужен PANDASCORE_TOKEN (бесплатно на app.pandascore.co)")
        self.f = fetcher or Fetcher(min_interval=1.0)
        self.h = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def past(self, pages: int = 20, days: int | None = None) -> list[Match]:
        out: list[Match] = []
        stop = datetime.now(timezone.utc) - timedelta(days=days) if days else None
        for page in range(1, pages + 1):
            try:
                data = self.f.get_json(f"{BASE}/matches/past", headers=self.h, params={
                    "sort": "-begin_at", "page[size]": 100, "page[number]": page, "filter[status]": "finished"})
            except Exception as e:  # лимит запросов и т.п. — сохраняем то, что успели
                if not out:
                    raise
                log.warning("PandaScore остановлен на странице %s: %s", page, e)
                break
            if not data:
                break
            batch = [m for m in map(parse_match, data) if m]
            out += batch
            log.info("PandaScore page %s: %s матчей", page, len(batch))
            if stop and batch and min(m.date for m in batch) < stop:
                break
        return [m for m in out if not stop or m.date >= stop]

    def upcoming(self, hours: float = 48) -> list[dict]:
        now = datetime.now(timezone.utc)
        data = self.f.get_json(f"{BASE}/matches/upcoming", headers=self.h, params={
            "sort": "begin_at", "page[size]": 100,
            "range[begin_at]": f"{now:%Y-%m-%dT%H:%M:%SZ},{now + timedelta(hours=hours):%Y-%m-%dT%H:%M:%SZ}"})
        return [u for u in map(parse_upcoming, data) if u]
