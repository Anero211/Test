"""Работа с коэффициентами: маржа, безмаржинальные вероятности, EV, Келли."""
from __future__ import annotations

import math


def margin(odds1: float, odds2: float) -> float:
    """Маржа букмекера: 1/k1 + 1/k2 - 1."""
    return 1 / odds1 + 1 / odds2 - 1


def fair_probs(odds1: float, odds2: float, method: str = "power") -> tuple[float, float]:
    """Вероятности исходов по коэффициентам с вычетом маржи.

    proportional — делим 1/k на сумму;
    power        — ищем n: (1/k1)^n + (1/k2)^n = 1 (учитывает, что БК сильнее
                   режет аутсайдера — «favourite-longshot bias»).
    """
    r1, r2 = 1 / odds1, 1 / odds2
    if method == "proportional" or abs(r1 + r2 - 1) < 1e-9:
        s = r1 + r2
        return r1 / s, r2 / s
    lo, hi = 0.5, 3.0
    for _ in range(80):
        n = (lo + hi) / 2
        if r1 ** n + r2 ** n > 1:
            lo = n
        else:
            hi = n
    n = (lo + hi) / 2
    p1 = r1 ** n
    return p1, 1 - p1


def expected_value(p: float, odds: float) -> float:
    """EV на 1 единицу ставки: P * k - 1."""
    return p * odds - 1


def kelly_fraction(p: float, odds: float, fraction: float = 0.25) -> float:
    """Доля банка по Келли (по умолчанию 1/4 Келли — стандартная осторожная версия)."""
    b = odds - 1
    if b <= 0:
        return 0.0
    f = (p * odds - 1) / b
    return max(0.0, f * fraction)


def implied_to_odds(p: float, book_margin: float = 0.0) -> float:
    return math.inf if p <= 0 else 1 / (p * (1 + book_margin))
