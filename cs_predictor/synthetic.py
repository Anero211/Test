"""Синтетическая «вселенная» CS-матчей для проверки кода без интернета.

ВАЖНО: результаты бэктеста на синтетике показывают только, что код работает
и модель умеет извлекать сигнал. Реальную точность даёт только бэктест на
настоящей истории (`collect` + `backtest`).
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .ratings import bo3_prob_from_maps
from .storage import MapResult, Match

MAPS = ["ancient", "anubis", "dust2", "inferno", "mirage", "nuke", "train"]


def generate(n_teams: int = 80, days: int = 540, matches_per_day: int = 14, seed: int = 7,
             book_noise: float = 0.15, book_margin: float = 0.065) -> list[Match]:
    rng = random.Random(seed)
    skill = {f"Team{i:02d}": rng.gauss(0, 130) for i in range(n_teams)}
    map_skill = {t: {m: rng.gauss(0, 45) for m in MAPS} for t in skill}
    teams = list(skill)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out: list[Match] = []
    mid = 0

    def p_map(a, b, m):
        d = skill[a] - skill[b] + map_skill[a][m] - map_skill[b][m]
        return 1 / (1 + 10 ** (-d / 400))

    for day in range(days):
        for t in teams:  # дрейф силы и редкие «смены состава»
            skill[t] += rng.gauss(0, 4)
            if rng.random() < 0.0015:
                skill[t] += rng.gauss(0, 90)
        ranked = sorted(teams, key=lambda t: skill[t] + rng.gauss(0, 60))
        for _ in range(matches_per_day):
            i = rng.randrange(len(ranked))
            j = min(max(i + rng.choice([-3, -2, -1, 1, 2, 3]), 0), len(ranked) - 1)
            if i == j:
                continue
            a, b = ranked[i], ranked[j]
            bo = 3 if rng.random() < 0.55 else 1
            pool = MAPS[:]
            if bo == 1:
                rng.shuffle(pool)
                maps = [pool[0]]
            else:  # вето по истинным силам карт
                left = {m: p_map(a, b, m) for m in pool}
                picks = []
                for k, act in enumerate(["ban", "ban", "pick", "pick", "ban", "ban"]):
                    a_turn = k % 2 == 0
                    choose_max = (act == "pick") == a_turn
                    m = max(left, key=left.get) if choose_max else min(left, key=left.get)
                    if act == "pick":
                        picks.append(m)
                    del left[m]
                maps = picks + list(left)
            results, s1, s2 = [], 0, 0
            need = bo // 2 + 1
            for m in maps:
                if s1 == need or s2 == need:
                    break
                w = 1 if rng.random() < p_map(a, b, m) else 2
                results.append(MapResult(m, w))
                s1 += w == 1
                s2 += w == 2
            # Коэффициенты «букмекера»: истинная вероятность + ошибка БК + маржа
            if bo == 1:
                true_p = sum(p_map(a, b, m) for m in MAPS) / len(MAPS)
            else:
                true_p = bo3_prob_from_maps(*(p_map(a, b, m) for m in maps[:3]))
            q = 1 / (1 + math.exp(-(math.log(true_p / (1 - true_p)) + rng.gauss(0, book_noise))))
            o1 = round(1 / (q * (1 + book_margin)), 2)
            o2 = round(1 / ((1 - q) * (1 + book_margin)), 2)
            date = start + timedelta(days=day, minutes=rng.randrange(10 * 60, 23 * 60))
            mid += 1
            out.append(Match(f"syn{mid}", date, a, b, s1, s2, bo, "Synthetic Cup", "synthetic",
                             results, max(o1, 1.01), max(o2, 1.01)))
    out.sort(key=lambda m: m.date)
    return out
