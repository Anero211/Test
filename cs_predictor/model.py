"""Итоговая вероятностная модель: признаки из рейтингового движка + логистическая калибровка.

Признаки (с точки зрения team1):
  elo_logit  — logit вероятности серии по Elo/маппулу/вето
  form       — разница «формы» (результат против ожидания в последних матчах)
  h2h        — личные встречи за год
  exp        — разница опыта (log числа матчей в базе)
  idle       — разница дней простоя

Коэффициенты подбираются логистической регрессией только на ПРОШЛЫХ матчах
(walk-forward), поэтому модель честно проверяется в бэктесте.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .ratings import EloParams, RatingEngine, logistic, logit
from .storage import Match

FEATURES = ["elo_logit", "form", "h2h", "exp", "idle"]


def build_features(engine: RatingEngine, a: str, b: str, best_of: int, at: datetime) -> np.ndarray:
    p = engine.p_series(a, b, best_of, at)
    return np.array([
        logit(p),
        engine.form_score(a) - engine.form_score(b),
        engine.h2h_score(a, b, at),
        math.log1p(engine.games[a]) - math.log1p(engine.games[b]),
        (engine.days_idle(a, at) - engine.days_idle(b, at)) / 30.0,
    ])


class LogisticModel:
    """Логистическая регрессия без свободного члена (данные симметризуются), L2, метод Ньютона."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        # До обучения — чистый Elo (коэффициент 1 у elo_logit, остальные 0)
        self.w = np.zeros(len(FEATURES))
        self.w[0] = 1.0
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 30) -> "LogisticModel":
        X = np.vstack([X, -X])
        y = np.concatenate([y, 1 - y])
        w = self.w.copy()
        prior = np.zeros_like(w)
        prior[0] = 1.0  # регуляризуем к «чистому Elo», а не к нулю
        for _ in range(iters):
            z = np.clip(X @ w, -35, 35)
            p = 1 / (1 + np.exp(-z))
            grad = X.T @ (p - y) + self.l2 * (w - prior)
            H = (X * (p * (1 - p))[:, None]).T @ X + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> float:
        return logistic(float(x @ self.w))


@dataclass
class Prediction:
    team1: str
    team2: str
    best_of: int
    p_elo: float        # вероятность победы team1 по чистому рейтингу
    p: float            # итоговая откалиброванная вероятность победы team1
    games1: int
    games2: int
    features: dict


class Predictor:
    """Обучается на всей истории и выдаёт прогнозы на будущие матчи."""

    def __init__(self, params: EloParams | None = None, l2: float = 1.0, burn_in: int = 300):
        self.engine = RatingEngine(params)
        self.model = LogisticModel(l2)
        self.burn_in = burn_in
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._seen = 0

    def observe(self, m: Match) -> np.ndarray | None:
        """Сначала фиксирует признаки ДО матча (для обучения), затем обновляет рейтинги."""
        x = None
        if m.winner != 0 and m.team1 != m.team2:
            x = build_features(self.engine, m.team1, m.team2, m.best_of, m.date)
            if self._seen >= self.burn_in:
                self._X.append(x)
                self._y.append(1.0 if m.winner == 1 else 0.0)
        self.engine.update(m)
        self._seen += 1
        return x

    def fit(self) -> None:
        if len(self._y) >= 50:
            self.model.fit(np.array(self._X), np.array(self._y))

    def train(self, matches: list[Match]) -> "Predictor":
        for m in sorted(matches, key=lambda m: m.date):
            self.observe(m)
        self.fit()
        return self

    def predict(self, a: str, b: str, best_of: int, at: datetime) -> Prediction:
        x = build_features(self.engine, a, b, best_of, at)
        return Prediction(
            team1=a, team2=b, best_of=best_of,
            p_elo=logistic(x[0]), p=self.model.predict(x),
            games1=self.engine.games[a], games2=self.engine.games[b],
            features=dict(zip(FEATURES, map(float, x))),
        )
