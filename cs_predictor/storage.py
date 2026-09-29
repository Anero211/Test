"""Модель данных матча и хранение истории в CSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .names import normalize

FIELDS = [
    "match_id", "date", "team1", "team2", "score1", "score2", "best_of",
    "event", "source", "maps", "odds1", "odds2",
]


@dataclass
class MapResult:
    name: str
    winner: int  # 1 или 2 (с точки зрения team1/team2 матча)


@dataclass
class Match:
    match_id: str
    date: datetime  # UTC
    team1: str
    team2: str
    score1: int  # выиграно карт (для Bo1 — 1/0)
    score2: int
    best_of: int
    event: str = ""
    source: str = ""
    maps: list[MapResult] = field(default_factory=list)
    odds1: float | None = None  # коэффициенты БК до матча (если известны) — для бэктеста ROI
    odds2: float | None = None

    @property
    def winner(self) -> int:
        return 1 if self.score1 > self.score2 else 2 if self.score2 > self.score1 else 0


def parse_dt(value: str | int | float | None) -> datetime | None:
    """Разбирает ISO-строку или unix-время (секунды/миллисекунды) в aware-UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().lstrip("-").isdigit()):
        ts = float(value)
        if ts > 1e11:  # миллисекунды
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _maps_to_str(maps: list[MapResult]) -> str:
    return ";".join(f"{m.name}:{m.winner}" for m in maps)


def _maps_from_str(s: str) -> list[MapResult]:
    out = []
    for part in (s or "").split(";"):
        if ":" in part:
            name, w = part.rsplit(":", 1)
            if w in ("1", "2"):
                out.append(MapResult(name.strip().lower(), int(w)))
    return out


def _float_or_none(v: str) -> float | None:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def load_matches(path: Path) -> list[Match]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt = parse_dt(row["date"])
            if dt is None:
                continue
            out.append(Match(
                match_id=row["match_id"], date=dt, team1=row["team1"], team2=row["team2"],
                score1=int(row["score1"]), score2=int(row["score2"]), best_of=int(row["best_of"]),
                event=row.get("event", ""), source=row.get("source", ""),
                maps=_maps_from_str(row.get("maps", "")),
                odds1=_float_or_none(row.get("odds1", "")), odds2=_float_or_none(row.get("odds2", "")),
            ))
    out.sort(key=lambda m: m.date)
    return out


def save_matches(path: Path, matches: Iterable[Match]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in sorted(matches, key=lambda m: m.date):
            w.writerow({
                "match_id": m.match_id, "date": m.date.isoformat(), "team1": m.team1, "team2": m.team2,
                "score1": m.score1, "score2": m.score2, "best_of": m.best_of, "event": m.event,
                "source": m.source, "maps": _maps_to_str(m.maps),
                "odds1": "" if m.odds1 is None else m.odds1, "odds2": "" if m.odds2 is None else m.odds2,
            })


def _dup_key(m: Match) -> tuple:
    pair = tuple(sorted((normalize(m.team1), normalize(m.team2))))
    return pair, m.date.date()


def merge_matches(existing: list[Match], new: list[Match]) -> list[Match]:
    """Объединяет историю из разных источников без дублей.

    Дубль = тот же match_id, либо матч из ДРУГОГО источника с той же парой команд
    в тот же/соседний день. Приоритет у записи с картами (более подробной).
    """
    by_id: dict[str, Match] = {m.match_id: m for m in existing}
    for m in new:
        cur = by_id.get(m.match_id)
        if cur is None or (not cur.maps and m.maps):
            by_id[m.match_id] = m
    kept: list[Match] = []
    index: dict[tuple, list[int]] = {}
    for m in sorted(by_id.values(), key=lambda m: m.date):
        pair, day = _dup_key(m)
        dup = None
        for d in (day, date.fromordinal(day.toordinal() - 1)):
            for i in index.get((pair, d), []):
                if kept[i].source != m.source:
                    dup = i
                    break
        if dup is None:
            index.setdefault((pair, day), []).append(len(kept))
            kept.append(m)
        elif not kept[dup].maps and m.maps:
            kept[dup] = m
    return kept
