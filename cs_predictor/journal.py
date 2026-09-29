"""Журнал ставок бота: каждая рекомендация записывается с коэффициентом на момент сигнала,
потом сама рассчитывается по результатам HLTV. Отсюда — реальный ROI стратегии.

Ставка считается «виртуальной»: бот не знает, поставил ли ты на самом деле. Журнал отвечает
на вопрос «зарабатывает ли стратегия, если ставить всё, что она советует».
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .storage import Match, parse_dt


@dataclass
class Bet:
    bet_id: str
    created_at: str      # когда пришёл сигнал (UTC ISO)
    start: str           # начало матча (UTC ISO, может быть пусто)
    team1: str           # канонические названия (как в истории HLTV)
    team2: str
    best_of: int
    pick: str            # на кого ставка (каноническое название)
    market: str          # «П1» / «П2»
    odds: float
    p_model: float
    ev: float
    stake: float         # в процентах банка
    status: str = "open"  # open / won / lost / void
    settled_at: str = ""
    score: str = ""
    profit: float = 0.0  # в процентах банка

    @property
    def key(self) -> str:
        return f"{self.start[:10] or self.created_at[:10]}|{self.team1}|{self.team2}|{self.pick}"


def load(path: Path) -> list[Bet]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(Bet(**{
                fl.name: (float(r[fl.name]) if fl.type in ("float", float) else
                          int(r[fl.name]) if fl.type in ("int", int) else r[fl.name])
                for fl in fields(Bet)}))
    return out


def save(path: Path, bets: list[Bet]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(Bet)])
        w.writeheader()
        for b in bets:
            w.writerow(asdict(b))


def add_if_new(bets: list[Bet], bet: Bet) -> bool:
    """Одна запись на матч и сторону: фиксируем коэффициент первого сигнала."""
    if any(b.key == bet.key for b in bets):
        return False
    bets.append(bet)
    return True


def settle(bets: list[Bet], history: list[Match], now: datetime | None = None) -> list[Bet]:
    """Рассчитывает открытые ставки по истории матчей. Возвращает только что рассчитанные."""
    now = now or datetime.now(timezone.utc)
    by_pair: dict[frozenset, list[Match]] = {}
    for m in history:
        by_pair.setdefault(frozenset((m.team1, m.team2)), []).append(m)
    done = []
    for b in bets:
        if b.status != "open":
            continue
        ref = parse_dt(b.start) or parse_dt(b.created_at)
        cands = [m for m in by_pair.get(frozenset((b.team1, b.team2)), [])
                 if ref - timedelta(hours=6) <= m.date <= ref + timedelta(hours=36)]
        if cands:
            m = min(cands, key=lambda m: abs((m.date - ref).total_seconds()))
            winner = m.team1 if m.winner == 1 else m.team2 if m.winner == 2 else None
            if winner is None:
                b.status, b.profit = "void", 0.0
            elif winner == b.pick:
                b.status, b.profit = "won", round(b.stake * (b.odds - 1), 4)
            else:
                b.status, b.profit = "lost", -b.stake
            s1, s2 = (m.score1, m.score2) if m.team1 == b.team1 else (m.score2, m.score1)
            b.score, b.settled_at = f"{s1}:{s2}", now.isoformat()
            done.append(b)
        elif ref < now - timedelta(days=4):  # результат так и не появился (перенос/отмена)
            b.status, b.settled_at = "void", now.isoformat()
            done.append(b)
    return done


def stats(bets: list[Bet], since: datetime | None = None) -> dict:
    closed = [b for b in bets if b.status in ("won", "lost")
              and (since is None or (parse_dt(b.settled_at) or since) >= since)]
    staked = sum(b.stake for b in closed)
    profit = sum(b.profit for b in closed)
    n = len(closed)
    return {
        "open": sum(1 for b in bets if b.status == "open"),
        "settled": n,
        "won": sum(1 for b in closed if b.status == "won"),
        "hit_rate": (sum(1 for b in closed if b.status == "won") / n) if n else float("nan"),
        "avg_odds": (sum(b.odds for b in closed) / n) if n else float("nan"),
        "expected_hit": (sum(b.p_model for b in closed) / n) if n else float("nan"),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked else float("nan"),
    }


def report(bets: list[Bet]) -> str:
    st = stats(bets)
    if st["settled"] == 0:
        return f"Журнал ставок: рассчитанных пока нет, открыто {st['open']}."
    lines = [
        f"Журнал ставок: рассчитано {st['settled']}, открыто {st['open']}",
        f"Угадано: {st['won']}/{st['settled']} = {st['hit_rate']:.1%} (модель ожидала {st['expected_hit']:.1%})",
        f"Средний кэф: {st['avg_odds']:.2f}",
        f"Поставлено: {st['staked']:.1f}% банка, прибыль: {st['profit']:+.2f}% банка, ROI: {st['roi']:+.1%}",
    ]
    if st["settled"] < 50:
        lines.append(f"⚠ Ставок меньше 50 — ROI пока в основном шум, выводы делать рано.")
    last = [b for b in bets if b.status in ("won", "lost")][-10:]
    if last:
        lines.append("Последние:")
        for b in last:
            lines.append(f"  {'✅' if b.status == 'won' else '❌'} {b.team1} – {b.team2} ({b.score}): "
                         f"{b.market} {b.pick} @ {b.odds:.2f} → {b.profit:+.2f}%")
    return "\n".join(lines)
