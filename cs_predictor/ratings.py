"""Рейтинговый движок: Elo по картам + поправки на маппул + симуляция вето Bo3.

Идея:
* каждая сыгранная карта — отдельная «партия» Elo (Bo3 2:1 = три обновления);
* у каждой команды есть смещение рейтинга на каждой карте (сильные/слабые карты);
* вероятность серии считается из вероятностей отдельных карт с учётом вето;
* после долгого простоя рейтинг частично откатывается к среднему (вероятная смена состава).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from .storage import Match


def logistic(x: float) -> float:
    if x < -35:
        return 0.0
    if x > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def series_win_prob(p: float, best_of: int) -> float:
    """Вероятность выиграть серию Bo-N при вероятности p выиграть каждую карту.

    Bo3: p^2 * (3 - 2p). Bo2 (возможна ничья) — вероятность выиграть 2:0.
    """
    if best_of <= 1:
        return p
    if best_of == 2:
        return p * p
    need = best_of // 2 + 1
    return sum(math.comb(need - 1 + j, j) * p ** need * (1 - p) ** j for j in range(need))


def bo3_prob_from_maps(p1: float, p2: float, p3: float) -> float:
    """Bo3 с разными вероятностями на пике 1, пике 2 и десайдере."""
    return p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3


@dataclass
class EloParams:
    init: float = 1500.0
    k: float = 28.0            # K-фактор общего рейтинга (за одну карту)
    k_new: float = 48.0        # повышенный K для команд с малым числом игр
    new_games: int = 15
    k_map: float = 10.0        # K для смещений по картам
    map_offset_cap: float = 150.0
    scale: float = 400.0
    inactivity_days: int = 45  # после такого простоя считаем, что состав мог поменяться
    inactivity_shrink: float = 0.3
    form_window: int = 10


class RatingEngine:
    def __init__(self, params: EloParams | None = None):
        self.p = params or EloParams()
        self.rating: dict[str, float] = {}
        self.map_off: dict[str, dict[str, float]] = defaultdict(dict)
        self.map_games: dict[str, Counter] = defaultdict(Counter)
        self.last_played: dict[str, datetime] = {}
        self.games: Counter = Counter()  # сыграно матчей
        # форма: (факт. доля карт - ожидаемая доля) в последних матчах
        self.form: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.p.form_window))
        self.h2h: dict[tuple[str, str], list[tuple[datetime, str]]] = defaultdict(list)
        self.map_counts: Counter = Counter()  # для автоопределения актуального маппула
        self.map_last_seen: dict[str, datetime] = {}

    # ---------- чтение состояния ----------
    def get_rating(self, team: str, at: datetime | None = None) -> float:
        r = self.rating.get(team, self.p.init)
        last = self.last_played.get(team)
        if at is not None and last is not None and (at - last).days > self.p.inactivity_days:
            r = self.p.init + (r - self.p.init) * (1 - self.p.inactivity_shrink)
        return r

    def p_map(self, a: str, b: str, map_name: str | None = None, at: datetime | None = None) -> float:
        diff = self.get_rating(a, at) - self.get_rating(b, at)
        if map_name:
            diff += self.map_off[a].get(map_name, 0.0) - self.map_off[b].get(map_name, 0.0)
        return 1.0 / (1.0 + 10 ** (-diff / self.p.scale))

    def active_pool(self, at: datetime | None = None, size: int = 7, recent_days: int = 120) -> list[str]:
        """Актуальный маппул = самые частые карты, игравшиеся за последние recent_days."""
        if at is None:
            at = max(self.map_last_seen.values(), default=None)
        maps = [m for m, c in self.map_counts.most_common()
                if at is None or (at - self.map_last_seen[m]) <= timedelta(days=recent_days)]
        return maps[:size]

    def has_map_data(self, team: str, min_maps: int = 8) -> bool:
        return sum(self.map_games[team].values()) >= min_maps

    def veto_bo3(self, a: str, b: str, at: datetime | None = None) -> float | None:
        """Симуляция вето Bo3 (бан-бан-пик-пик-бан-бан-десайдер). None, если данных по картам мало."""
        pool = self.active_pool(at)
        if len(pool) < 7 or not (self.has_map_data(a) and self.has_map_data(b)):
            return None
        probs = {m: self.p_map(a, b, m, at) for m in pool}

        def run(first_is_a: bool) -> float:
            left = dict(probs)
            order = ["ban", "ban", "pick", "pick", "ban", "ban"]
            picks = []
            for i, action in enumerate(order):
                a_turn = (i % 2 == 0) == first_is_a
                if action == "ban":  # банит худшую для себя карту
                    m = min(left, key=left.get) if a_turn else max(left, key=left.get)
                else:  # пикает лучшую для себя
                    m = max(left, key=left.get) if a_turn else min(left, key=left.get)
                if action == "pick":
                    picks.append(left[m])
                del left[m]
            decider = next(iter(left.values()))
            return bo3_prob_from_maps(picks[0], picks[1], decider)

        return 0.5 * (run(True) + run(False))

    def p_series(self, a: str, b: str, best_of: int, at: datetime | None = None) -> float:
        base = series_win_prob(self.p_map(a, b, None, at), best_of)
        if best_of == 3:
            v = self.veto_bo3(a, b, at)
            if v is not None:
                return 0.5 * base + 0.5 * v
        return base

    def form_score(self, team: str) -> float:
        f = self.form.get(team)
        if not f:
            return 0.0
        w = [0.85 ** i for i in range(len(f))][::-1]  # свежие матчи весят больше
        return sum(x * wi for x, wi in zip(f, w)) / sum(w)

    def h2h_score(self, a: str, b: str, at: datetime, days: int = 365) -> float:
        key = tuple(sorted((a, b)))
        rec = [w for d, w in self.h2h.get(key, []) if (at - d).days <= days]
        if not rec:
            return 0.0
        wa = sum(1 for w in rec if w == a)
        return (wa - (len(rec) - wa)) / (len(rec) + 2)

    def days_idle(self, team: str, at: datetime) -> float:
        last = self.last_played.get(team)
        return 60.0 if last is None else min((at - last).total_seconds() / 86400, 60.0)

    # ---------- обновление ----------
    def _k(self, team: str) -> float:
        return self.p.k_new if self.games[team] < self.p.new_games else self.p.k

    def _apply_inactivity(self, team: str, at: datetime) -> None:
        if team in self.rating:
            self.rating[team] = self.get_rating(team, at)

    def update(self, m: Match) -> None:
        a, b = m.team1, m.team2
        if m.winner == 0 or a == b:
            return
        for t in (a, b):
            self._apply_inactivity(t, m.date)
            self.rating.setdefault(t, self.p.init)

        exp_share = self.p_map(a, b)  # ожидаемая доля карт до матча (для «формы»)

        # Карточные результаты: если известны карты — по ним, иначе восстанавливаем по счёту
        if m.maps:
            games = [(mr.name, mr.winner) for mr in m.maps]
        else:
            games = [(None, 1)] * m.score1 + [(None, 2)] * m.score2
            if m.best_of == 1 and not games:
                games = [(None, m.winner)]

        for map_name, winner in games:
            s = 1.0 if winner == 1 else 0.0
            e = self.p_map(a, b)
            ka, kb = self._k(a), self._k(b)
            self.rating[a] += ka * (s - e)
            self.rating[b] -= kb * (s - e)
            if map_name:
                em = self.p_map(a, b, map_name)
                d = self.p.k_map * (s - em)
                cap = self.p.map_offset_cap
                self.map_off[a][map_name] = max(-cap, min(cap, self.map_off[a].get(map_name, 0.0) + d))
                self.map_off[b][map_name] = max(-cap, min(cap, self.map_off[b].get(map_name, 0.0) - d))
                self.map_games[a][map_name] += 1
                self.map_games[b][map_name] += 1
                self.map_counts[map_name] += 1
                self.map_last_seen[map_name] = m.date

        total = max(m.score1 + m.score2, 1)
        share_a = m.score1 / total if m.best_of > 1 else float(m.winner == 1)
        self.form[a].append(share_a - exp_share)
        self.form[b].append((1 - share_a) - (1 - exp_share))
        self.h2h[tuple(sorted((a, b)))].append((m.date, a if m.winner == 1 else b))
        for t in (a, b):
            self.games[t] += 1
            self.last_played[t] = m.date
