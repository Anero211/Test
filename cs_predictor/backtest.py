"""Walk-forward бэктест: модель на каждом матче знает только то, что было ДО него."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import Settings
from .model import Predictor
from .odds import expected_value, fair_probs
from .ratings import EloParams
from .storage import Match, parse_dt


@dataclass
class BTRow:
    date: str
    event: str
    team1: str
    team2: str
    best_of: int
    p_elo: float
    p: float
    win1: int
    games1: int
    games2: int
    odds1: float | None
    odds2: float | None


def run_backtest(matches: list[Match], since: datetime | None = None, until: datetime | None = None,
                 burn_in: int = 300, refit_every: int = 200, params: EloParams | None = None) -> list[BTRow]:
    pred = Predictor(params, burn_in=burn_in)
    rows: list[BTRow] = []
    since_fit = 0
    for m in sorted(matches, key=lambda m: m.date):
        in_window = (since is None or m.date >= since) and (until is None or m.date < until)
        if in_window and pred._seen >= burn_in and m.winner != 0 and m.team1 != m.team2:
            pr = pred.predict(m.team1, m.team2, m.best_of, m.date)
            rows.append(BTRow(
                date=m.date.isoformat(), event=m.event, team1=m.team1, team2=m.team2, best_of=m.best_of,
                p_elo=pr.p_elo, p=pr.p, win1=int(m.winner == 1), games1=pr.games1, games2=pr.games2,
                odds1=m.odds1, odds2=m.odds2,
            ))
        pred.observe(m)
        since_fit += 1
        if pred._seen >= burn_in and since_fit >= refit_every:
            pred.fit()
            since_fit = 0
    return rows


# ---------------- метрики ----------------

def _scores(rows: list[BTRow], attr: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "logloss": float("nan"), "brier": float("nan")}
    acc = ll = br = 0.0
    for r in rows:
        p = min(max(getattr(r, attr), 1e-6), 1 - 1e-6)
        acc += (p >= 0.5) == bool(r.win1)
        ll += -(r.win1 * math.log(p) + (1 - r.win1) * math.log(1 - p))
        br += (p - r.win1) ** 2
    return {"n": n, "acc": acc / n, "logloss": ll / n, "brier": br / n}


def _pick(r: BTRow, attr: str = "p") -> tuple[float, bool]:
    """(уверенность в фаворите модели, угадал ли)."""
    p = getattr(r, attr)
    return max(p, 1 - p), (p >= 0.5) == bool(r.win1)


def calibration(rows: list[BTRow], edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.01)) -> list[dict]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [_pick(r) for r in rows if lo <= _pick(r)[0] < hi]
        if sel:
            out.append({"bucket": f"{lo:.0%}-{min(hi, 1):.0%}", "n": len(sel),
                        "predicted": sum(c for c, _ in sel) / len(sel),
                        "actual": sum(h for _, h in sel) / len(sel)})
    return out


def confidence_table(rows: list[BTRow], thresholds=(0.55, 0.6, 0.65, 0.7, 0.75, 0.8)) -> list[dict]:
    out = []
    for t in thresholds:
        sel = [_pick(r) for r in rows if _pick(r)[0] >= t]
        out.append({"threshold": t, "n": len(sel), "share": len(sel) / max(len(rows), 1),
                    "acc": (sum(h for _, h in sel) / len(sel)) if sel else float("nan")})
    return out


def top_daily(rows: list[BTRow], top_n: int = 3, best_of: int | None = 3) -> dict:
    """Точность «N самых уверенных прогнозов дня» — то, как скрипт используется в жизни."""
    by_day = defaultdict(list)
    for r in rows:
        if best_of is None or r.best_of == best_of:
            by_day[r.date[:10]].append(r)
    picks = []
    for day_rows in by_day.values():
        picks += sorted(day_rows, key=lambda r: -_pick(r)[0])[:top_n]
    hits = [_pick(r)[1] for r in picks]
    return {"days": len(by_day), "n": len(hits), "acc": (sum(hits) / len(hits)) if hits else float("nan")}


def betting_sim(rows: list[BTRow], s: Settings) -> dict:
    """Симуляция стратегии из ТЗ на матчах, где известны коэффициенты."""
    bets = []
    with_odds = [r for r in rows if r.odds1 and r.odds2]
    for r in with_odds:
        if r.best_of not in s.allowed_formats or min(r.games1, r.games2) < s.min_team_games:
            continue
        for side, p, k, won in ((1, r.p, r.odds1, r.win1 == 1), (2, 1 - r.p, r.odds2, r.win1 == 0)):
            ev = expected_value(p, k)
            if s.min_odds <= k <= s.max_odds and s.min_ev <= ev <= s.max_ev:
                bets.append((r.date, k, won, ev))
    bets.sort()
    bank = peak = dd = 0.0
    for _, k, won, _ in bets:
        bank += (k - 1) if won else -1
        peak = max(peak, bank)
        dd = max(dd, peak - bank)
    n = len(bets)
    # Для сравнения: как часто угадывает фаворит БК на тех же матчах
    fav_hits = [((fair_probs(r.odds1, r.odds2)[0] >= 0.5) == bool(r.win1)) for r in with_odds]
    return {
        "matches_with_odds": len(with_odds),
        "bets": n,
        "hit_rate": (sum(b[2] for b in bets) / n) if n else float("nan"),
        "avg_odds": (sum(b[1] for b in bets) / n) if n else float("nan"),
        "avg_ev_model": (sum(b[3] for b in bets) / n) if n else float("nan"),
        "profit_units": bank,
        "roi": bank / n if n else float("nan"),
        "max_drawdown_units": dd,
        "bookmaker_fav_acc": (sum(fav_hits) / len(fav_hits)) if fav_hits else float("nan"),
        "model_acc_same_matches": _scores(with_odds, "p")["acc"],
    }


def _fmt(x, pct=False) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "—"
    return f"{x:.1%}" if pct else (f"{x:.3f}" if isinstance(x, float) else str(x))


def report(rows: list[BTRow], s: Settings, title: str = "Бэктест") -> str:
    est = [r for r in rows if min(r.games1, r.games2) >= s.min_team_games]
    lines = [f"# {title}", ""]
    if rows:
        lines.append(f"Период: {rows[0].date[:10]} — {rows[-1].date[:10]}, матчей в тесте: {len(rows)} "
                     f"(обе команды с историей ≥{s.min_team_games} матчей: {len(est)})")
    lines += ["", "## Общая точность (угадан победитель)", "",
              "| Выборка | N | Точность итог. модели | Точность чистого Elo | LogLoss | Brier |",
              "|---|---|---|---|---|---|"]
    for name, sub in (("Все", rows), ("Опытные команды", est),
                      ("Bo3", [r for r in est if r.best_of == 3]), ("Bo1", [r for r in est if r.best_of == 1])):
        a, b = _scores(sub, "p"), _scores(sub, "p_elo")
        lines.append(f"| {name} | {a['n']} | {_fmt(a['acc'], True)} | {_fmt(b['acc'], True)} | "
                     f"{_fmt(a['logloss'])} | {_fmt(a['brier'])} |")
    lines += ["", "(угадывание наугад = 50%, LogLoss наугад = 0.693)", "",
              "## Точность в зависимости от уверенности модели (опытные команды)", "",
              "| Уверенность ≥ | Матчей | Доля от всех | Точность |", "|---|---|---|---|"]
    for row in confidence_table(est):
        lines.append(f"| {row['threshold']:.0%} | {row['n']} | {_fmt(row['share'], True)} | {_fmt(row['acc'], True)} |")
    lines += ["", "## Калибровка (обещанная вероятность vs факт)", "",
              "| Корзина | N | Модель обещала | Реально угадано |", "|---|---|---|---|"]
    for row in calibration(est):
        lines.append(f"| {row['bucket']} | {row['n']} | {_fmt(row['predicted'], True)} | {_fmt(row['actual'], True)} |")
    lines += ["", "## «Самые вероятные матчи дня» (Bo3, опытные команды)", ""]
    for n in (1, 3, 5):
        t = top_daily(est, n)
        lines.append(f"- топ-{n} в день: {t['n']} прогнозов за {t['days']} дней, точность {_fmt(t['acc'], True)}")
    sim = betting_sim(rows, s)
    lines += ["", f"## Симуляция ставок по фильтру (Bo3, кэф {s.min_odds}–{s.max_odds}, EV ≥ {s.min_ev:.0%})", ""]
    if sim["matches_with_odds"] == 0:
        lines.append("Нет исторических коэффициентов — ROI не считается. Запускай `watch`/`predict`: "
                     "скрипт копит снимки линии BetBoom в data/odds_log.csv, потом "
                     "`backtest --odds-log` посчитает реальную доходность стратегии.")
    else:
        for k, v in sim.items():
            pct = k in ("hit_rate", "roi", "avg_ev_model", "bookmaker_fav_acc", "model_acc_same_matches")
            lines.append(f"- {k}: {_fmt(v, pct)}")
    return "\n".join(lines) + "\n"


def save_rows(rows: list[BTRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(BTRow.__dataclass_fields__))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# ---------------- реальные коэффициенты из накопленного лога BetBoom ----------------

def attach_odds_log(matches: list[Match], odds_log: Path, matcher) -> int:
    """Проставляет матчам истории коэффициенты BetBoom из data/odds_log.csv (последний снимок до начала)."""
    if not odds_log.exists():
        return 0
    snaps: dict[frozenset, list[tuple]] = defaultdict(list)
    with odds_log.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = matcher.match(r["team1"])[0], matcher.match(r["team2"])[0]
            cap = parse_dt(r["captured_at"])
            if a and b and a != b and cap:
                snaps[frozenset((a, b))].append((cap, parse_dt(r["start"]), a, float(r["odds1"]), float(r["odds2"])))
    n = 0
    for m in matches:
        best = None
        for cap, start, a, o1, o2 in snaps.get(frozenset((m.team1, m.team2)), []):
            ref = start or cap
            if cap <= m.date + timedelta(minutes=5) and abs((ref - m.date).total_seconds()) <= 8 * 3600:
                if best is None or cap > best[0]:
                    best = (cap, a, o1, o2)
        if best:
            _, a, o1, o2 = best
            m.odds1, m.odds2 = (o1, o2) if a == m.team1 else (o2, o1)
            n += 1
    return n


# ---------------- подбор параметров Elo на реальной истории ----------------

def tune(matches: list[Match], s: Settings, burn_in: int = 300) -> tuple[EloParams, list[dict]]:
    """Перебор ключевых параметров; критерий — LogLoss walk-forward на опытных командах."""
    results = []
    for k in (20, 28, 36):
        for k_map in (0.0, 8.0, 14.0):
            for shrink in (0.15, 0.35):
                params = EloParams(k=k, k_map=k_map, inactivity_shrink=shrink)
                rows = [r for r in run_backtest(matches, burn_in=burn_in, params=params)
                        if min(r.games1, r.games2) >= s.min_team_games]
                sc = _scores(rows, "p")
                results.append({"k": k, "k_map": k_map, "inactivity_shrink": shrink, **sc})
    results.sort(key=lambda r: r["logloss"])
    b = results[0]
    return EloParams(k=b["k"], k_map=b["k_map"], inactivity_shrink=b["inactivity_shrink"]), results


def save_params(p: EloParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(p), indent=1))


def load_params(path: Path) -> EloParams:
    if path.exists():
        return EloParams(**json.loads(path.read_text()))
    return EloParams()
