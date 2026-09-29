"""Стратегия по предстоящему матчу: вероятности исходов и карт, справедливые коэффициенты,
минимальный коэффициент, с которого ставка выгодна, и уровень риска.

Букмекерские коэффициенты здесь не нужны: скрипт говорит «ставить на П1, если дают ≥ 1.62»,
а сравнить с линией своей БК можно глазами.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .model import Prediction, implied_map_prob

# Уровни риска по вероятности исхода
RISK_LEVELS = [(0.70, "низкий"), (0.55, "средний"), (0.40, "высокий"), (0.0, "очень высокий")]


def risk_of(p: float) -> str:
    return next(name for lo, name in RISK_LEVELS if p >= lo)


@dataclass
class Market:
    kind: str      # win / score / total / handicap
    label: str     # «П1 NAVI», «Счёт 2:0 NAVI», «Тотал карт Б 2.5», «Фора G2 +1.5»
    p: float
    fair_odds: float
    min_odds: float  # с этого коэффициента ставка даёт EV ≥ запаса
    risk: str


@dataclass
class MatchStrategy:
    start: datetime | None
    event: str
    team1: str
    team2: str
    best_of: int
    p1: float
    games1: int
    games2: int
    scores: dict[tuple[int, int], float]
    markets: list[Market] = field(default_factory=list)
    picks: list[tuple[str, Market]] = field(default_factory=list)  # (роль, рынок)
    note: str = ""

    @property
    def favorite(self) -> str:
        return self.team1 if self.p1 >= 0.5 else self.team2

    @property
    def fav_p(self) -> float:
        return max(self.p1, 1 - self.p1)

    @property
    def likely_score(self) -> tuple[tuple[int, int], float]:
        return max(self.scores.items(), key=lambda kv: kv[1])

    def market(self, label_prefix: str) -> Market | None:
        return next((m for m in self.markets if m.label.startswith(label_prefix)), None)


def score_distribution(p1: float, best_of: int, p3: float | None = None) -> dict[tuple[int, int], float]:
    """Вероятности точного счёта серии с точки зрения team1."""
    if best_of <= 1 or best_of == 2:
        return {(1, 0): p1, (0, 1): 1 - p1}
    q = implied_map_prob(p1, best_of)
    if best_of == 3:
        if p3 is None:
            p3 = 2 * q * (1 - q)
        # Доля 2:1 внутри «трёхкартовых» матчей делится пропорционально силе на карте,
        # и не может превысить вероятность победы каждой стороны
        p3 = min(p3, p1 / q if q > 0 else 0, (1 - p1) / (1 - q) if q < 1 else 0)
        return {(2, 0): p1 - p3 * q, (2, 1): p3 * q, (1, 2): p3 * (1 - q), (0, 2): 1 - p1 - p3 * (1 - q)}
    need = best_of // 2 + 1  # Bo5 и больше: карты независимы (данных для поправки мало)
    out = {}
    for k in range(need):
        out[(need, k)] = math.comb(need - 1 + k, k) * q ** need * (1 - q) ** k
        out[(k, need)] = math.comb(need - 1 + k, k) * (1 - q) ** need * q ** k
    return out


def _mk(kind: str, label: str, p: float, margin: float) -> Market:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return Market(kind, label, p, 1 / p, (1 + margin) / p, risk_of(p))


def build_strategy(pred: Prediction, start: datetime | None, event: str, team1: str, team2: str,
                   margin: float = 0.07, extra_margin_new: float = 0.05, new_games: int = 30) -> MatchStrategy:
    """team1/team2 — названия для вывода (как на HLTV).

    margin — запас прочности: минимальный кэф = (1 + margin) / P. Для команд с короткой историей
    запас увеличивается на extra_margin_new — модель там менее уверена.
    """
    if min(pred.games1, pred.games2) < new_games:
        margin += extra_margin_new
    bo = pred.best_of
    p1 = pred.p
    sc = score_distribution(p1, bo, pred.p3)
    st = MatchStrategy(start, event, team1, team2, bo, p1, pred.games1, pred.games2, sc)
    m = st.markets
    m.append(_mk("win", f"П1 {team1}", p1, margin))
    m.append(_mk("win", f"П2 {team2}", 1 - p1, margin))

    if bo >= 3:
        need = bo // 2 + 1
        for (a, b), p in sorted(sc.items(), key=lambda kv: -kv[1]):
            who = team1 if a > b else team2
            m.append(_mk("score", f"Счёт {max(a, b)}:{min(a, b)} {who}", p, margin))
        max_maps = bo
        for line in [x + 0.5 for x in range(need, max_maps)]:
            over = sum(p for (a, b), p in sc.items() if a + b > line)
            m.append(_mk("total", f"Тотал карт Б {line}", over, margin))
            m.append(_mk("total", f"Тотал карт М {line}", 1 - over, margin))
        # Форы по картам ±1.5: «-1.5» = выиграть с разницей ≥2 карт
        t1_big = sum(p for (a, b), p in sc.items() if a - b >= 2)
        t2_big = sum(p for (a, b), p in sc.items() if b - a >= 2)
        m.append(_mk("handicap", f"Фора {team1} -1.5", t1_big, margin))
        m.append(_mk("handicap", f"Фора {team2} +1.5", 1 - t1_big, margin))
        m.append(_mk("handicap", f"Фора {team2} -1.5", t2_big, margin))
        m.append(_mk("handicap", f"Фора {team1} +1.5", 1 - t2_big, margin))

    _choose_picks(st)
    if min(pred.games1, pred.games2) < new_games:
        st.note = "мало истории у одной из команд — запас по кэфу увеличен"
    return st


def _choose_picks(st: MatchStrategy) -> None:
    fav, dog = (st.team1, st.team2) if st.p1 >= 0.5 else (st.team2, st.team1)
    picks = [("основная", st.market(f"П{1 if fav == st.team1 else 2} {fav}"))]
    if st.best_of >= 3:
        need = st.best_of // 2 + 1
        dry = st.market(f"Счёт {need}:0 {fav}")
        if dry:
            picks.append(("рискованная", dry))
        totals = [mk for mk in st.markets if mk.kind == "total" and mk.label.endswith(f"{need}.5")]
        if totals:
            picks.append(("по картам", max(totals, key=lambda mk: mk.p)))
        cover = st.market(f"Фора {dog} +1.5")
        if cover:
            picks.append(("страховка андердога", cover))
    st.picks = [(role, mk) for role, mk in picks if mk is not None]


# ---------------- вывод ----------------

def _tz():
    import os
    from zoneinfo import ZoneInfo
    return ZoneInfo(os.environ.get("DISPLAY_TZ", "Europe/Moscow"))


def fmt_time(dt: datetime | None) -> str:
    return dt.astimezone(_tz()).strftime("%d.%m %H:%M") if dt else "—"


def _score_str(st: MatchStrategy) -> str:
    (a, b), p = st.likely_score
    return f"{a}:{b} ({p:.0%})"


def summary_table(items: list[MatchStrategy]) -> str:
    head = (f"{'#':>2}  {'Время':<11} {'Матч':<40} {'Bo':<3} {'Фаворит':<20} {'P':>5} "
            f"{'Справ.':>6} {'Ставить от':>10}  {'Риск':<13} {'Вер. счёт':<10} {'3 карты':>7}")
    lines = [head, "-" * len(head)]
    for i, st in enumerate(items, 1):
        main = st.picks[0][1]
        over = st.market("Тотал карт Б 2.5")
        lines.append(
            f"{i:>2}  {fmt_time(st.start):<11} {(st.team1 + ' – ' + st.team2)[:40]:<40} {st.best_of:<3} "
            f"{st.favorite[:20]:<20} {st.fav_p:>5.0%} {main.fair_odds:>6.2f} {main.min_odds:>10.2f}  "
            f"{main.risk:<13} {_score_str(st) if st.best_of > 1 else '—':<10} "
            f"{(f'{over.p:.0%}' if over else '—'):>7}")
    return "\n".join(lines)


def match_card(st: MatchStrategy, n: int | None = None, all_markets: bool = True) -> str:
    title = f"{n}. " if n else ""
    lines = [f"━━ {title}{st.team1} vs {st.team2} — Bo{st.best_of}, {fmt_time(st.start)} МСК, {st.event}",
             f"   История в базе: {st.games1} / {st.games2} матчей"
             + (f"   ⚠ {st.note}" if st.note else "")]
    if st.best_of > 1:
        dist = ", ".join(f"{a}:{b} {p:.0%}" for (a, b), p in sorted(st.scores.items(), key=lambda kv: -kv[1]))
        lines.append(f"   Счёт (за {st.team1}): {dist}")
    lines.append("   Стратегия:")
    for role, mk in st.picks:
        lines.append(f"     • {role + ':':<21} {mk.label[:34]:<34} {mk.p:>6.1%}  справ. {mk.fair_odds:>5.2f}  "
                     f"ставить от {mk.min_odds:>5.2f}  риск {mk.risk}")
    if st.best_of == 3:
        lines.append("   ℹ Тотал карт почти не зависит от силы команд (в истории ~40–45% матчей идут на 3 карты) —"
                     " брать только по кэфу не ниже указанного.")
    if all_markets and st.best_of > 1:
        lines.append("   Все рынки:")
        for mk in sorted(st.markets, key=lambda mk: -mk.p):
            if 0.05 <= mk.p <= 0.97:
                lines.append(f"       {mk.label[:34]:<34} {mk.p:>6.1%}  справ. {mk.fair_odds:>5.2f}  от {mk.min_odds:>5.2f}")
    return "\n".join(lines)


def telegram_text(st: MatchStrategy) -> str:
    import html
    e = html.escape
    out = [f"🎯 <b>{e(st.team1)} vs {e(st.team2)}</b> (Bo{st.best_of})", f"🕒 {fmt_time(st.start)} МСК · {e(st.event)}"]
    for role, mk in st.picks:
        out.append(f"• {role}: <b>{e(mk.label)}</b> — {mk.p:.0%}, ставить от <b>{mk.min_odds:.2f}</b> ({mk.risk})")
    return "\n".join(out)
