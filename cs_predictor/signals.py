"""Сведение линии BetBoom с прогнозами модели: EV, фильтр сигналов, списки на ближайшие 1–2 дня."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings
from .model import Predictor
from .names import TeamMatcher, normalize
from .odds import expected_value, fair_probs, kelly_fraction, margin
from .sources.betboom import OddsEvent


@dataclass
class Row:
    start: str
    tournament: str
    team1: str            # как в BetBoom
    team2: str
    model_team1: str      # как в базе истории ("" — не найдена)
    model_team2: str
    best_of: int | None
    odds1: float
    odds2: float
    margin: float
    book_p1: float        # вероятность по кэфам BetBoom без маржи
    p1: float             # вероятность модели
    ev1: float
    ev2: float
    pick: str             # команда-фаворит модели
    pick_p: float
    pick_odds: float
    pick_ev: float
    kelly: float          # доля банка (1/4 Келли) на pick
    games1: int
    games2: int
    signal: bool
    reason: str           # почему не сигнал


def _find_format(ev: OddsEvent, schedule: list[dict], matcher: TeamMatcher) -> int | None:
    """Формат матча: из линии BetBoom, иначе из расписания HLTV/PandaScore."""
    if ev.best_of:
        return ev.best_of

    def key(t1: str, t2: str) -> frozenset:
        return frozenset(matcher.match(t)[0] or normalize(t) for t in (t1, t2))

    target = key(ev.team1, ev.team2)
    for item in schedule:
        if item.get("best_of") and key(item["team1"], item["team2"]) == target:
            return int(item["best_of"])
    return None


def evaluate(line: list[OddsEvent], predictor: Predictor, matcher: TeamMatcher, schedule: list[dict],
             s: Settings, now: datetime | None = None) -> list[Row]:
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=s.horizon_hours)
    rows = []
    for ev in line:
        if ev.start and not (now - timedelta(minutes=10) <= ev.start <= horizon):
            continue
        a, _ = matcher.match(ev.team1)
        b, _ = matcher.match(ev.team2)
        bo = _find_format(ev, schedule, matcher)
        at = ev.start or now
        if a and b and a != b:
            pr = predictor.predict(a, b, bo or 3, at)
            p1, g1, g2 = pr.p, pr.games1, pr.games2
        else:
            p1, g1, g2 = float("nan"), 0, 0
        bp1, _ = fair_probs(ev.odds1, ev.odds2)
        ev1, ev2 = expected_value(p1, ev.odds1), expected_value(1 - p1, ev.odds2)
        known = p1 == p1  # не NaN
        sides = [(p1, ev.odds1, ev1, ev.team1), (1 - p1, ev.odds2, ev2, ev.team2)]

        common = []
        if not known:
            common.append("команда не найдена в истории")
        if bo not in s.allowed_formats:
            common.append(f"формат Bo{bo or '?'}")
        if known and min(g1, g2) < s.min_team_games:
            common.append("мало истории")
        # Ставка может быть и на андердога модели, если у него EV выше порога
        in_range = [sd for sd in sides if s.min_odds <= sd[1] <= s.max_odds]
        good = [sd for sd in in_range if known and sd[2] >= s.min_ev]
        if good:
            pick = max(good, key=lambda sd: sd[2])
            reasons = list(common)
        else:
            pick = max(sides, key=lambda sd: sd[2]) if known else sides[0]
            reasons = list(common)
            if not s.min_odds <= pick[1] <= s.max_odds:
                reasons.append("кэф вне диапазона")
            if known and pick[2] < s.min_ev:
                reasons.append("EV ниже порога")
        if known and pick[2] > s.max_ev:
            # Огромный «перевес» почти всегда означает, что модель чего-то не знает (замены, форма)
            reasons.append(f"EV>{s.max_ev:.0%}: подозрительно, проверь составы")
        pick_p, pick_odds, pick_ev, pick_team = pick
        rows.append(Row(
            start=ev.start.isoformat() if ev.start else "", tournament=ev.tournament,
            team1=ev.team1, team2=ev.team2, model_team1=a or "", model_team2=b or "", best_of=bo,
            odds1=ev.odds1, odds2=ev.odds2, margin=margin(ev.odds1, ev.odds2), book_p1=bp1, p1=p1,
            ev1=ev1, ev2=ev2, pick=pick_team, pick_p=pick_p, pick_odds=pick_odds, pick_ev=pick_ev,
            kelly=kelly_fraction(pick_p, pick_odds) if known else 0.0,
            games1=g1, games2=g2, signal=not reasons, reason=", ".join(reasons),
        ))
    rows.sort(key=lambda r: (not r.signal, -(r.pick_ev if r.pick_ev == r.pick_ev else -9)))
    return rows


def most_probable(rows: list[Row], n: int = 10, only_bo3: bool = True) -> list[Row]:
    """Самые уверенные прогнозы модели на горизонте (вне зависимости от EV) — ставка на фаворита модели."""
    out = []
    for r in rows:
        if r.p1 != r.p1 or min(r.games1, r.games2) == 0 or (only_bo3 and r.best_of != 3):
            continue
        fav1 = r.p1 >= 0.5
        p, k = (r.p1, r.odds1) if fav1 else (1 - r.p1, r.odds2)
        team = r.team1 if fav1 else r.team2
        out.append(replace(r, pick=team, pick_p=p, pick_odds=k, pick_ev=r.ev1 if fav1 else r.ev2,
                           kelly=kelly_fraction(p, k), signal=r.signal and r.pick == team,
                           reason=f"сигнал на {r.pick}" if r.signal and r.pick != team else "—"))
    return sorted(out, key=lambda r: -r.pick_p)[:n]


def format_table(rows: list[Row], title: str) -> str:
    if not rows:
        return f"{title}: нет матчей\n"
    lines = [title, f"{'Начало (UTC)':<16} {'Матч':<38} {'Bo':<3} {'К1':>5} {'К2':>5} {'P1 мод':>7} "
                    f"{'P1 БК':>6} {'Ставка':<18} {'P':>5} {'Кэф':>5} {'EV':>6} {'Келли':>6}  Примечание"]
    for r in rows:
        match = f"{r.team1} vs {r.team2}"[:38]
        p1 = f"{r.p1:.1%}" if r.p1 == r.p1 else "—"
        pp = f"{r.pick_p:.0%}" if r.pick_p == r.pick_p else "—"
        evs = f"{r.pick_ev:+.1%}" if r.pick_ev == r.pick_ev else "—"
        lines.append(f"{r.start[:16].replace('T', ' '):<16} {match:<38} {r.best_of or '?':<3} {r.odds1:>5.2f} "
                     f"{r.odds2:>5.2f} {p1:>7} {r.book_p1:>6.1%} {r.pick[:18]:<18} {pp:>5} {r.pick_odds:>5.2f} "
                     f"{evs:>6} {r.kelly:>6.1%}  {'✅ СИГНАЛ' if r.signal else r.reason}")
    return "\n".join(lines) + "\n"


def save_rows(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def log_odds(line: list[OddsEvent], path: Path, now: datetime | None = None) -> None:
    """Копит снимки линии BetBoom — потом по ним считается реальный ROI стратегии в бэктесте."""
    now = now or datetime.now(timezone.utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["captured_at", "start", "team1", "team2", "odds1", "odds2", "best_of", "tournament"])
        for ev in line:
            w.writerow([now.isoformat(), ev.start.isoformat() if ev.start else "", ev.team1, ev.team2,
                        ev.odds1, ev.odds2, ev.best_of or "", ev.tournament])
