from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from cs_predictor.backtest import attach_odds_log, report, run_backtest, top_daily
from cs_predictor.config import Settings
from cs_predictor.model import LogisticModel, Predictor
from cs_predictor.names import TeamMatcher, normalize
from cs_predictor.odds import expected_value, fair_probs, kelly_fraction, margin
from cs_predictor.ratings import RatingEngine, bo3_prob_from_maps, series_win_prob
from cs_predictor.signals import evaluate, log_odds, most_probable
from cs_predictor.sources.betboom import OddsEvent, extract_events, from_csv
from cs_predictor.sources.hltv import parse_match_maps, parse_results_page, parse_upcoming_page
from cs_predictor.storage import MapResult, Match, load_matches, merge_matches, save_matches
from cs_predictor.synthetic import generate

FIX = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------- математика ----------

def test_series_prob():
    assert series_win_prob(0.6, 1) == pytest.approx(0.6)
    assert series_win_prob(0.6, 3) == pytest.approx(0.6 ** 2 * (3 - 2 * 0.6))
    assert series_win_prob(0.5, 5) == pytest.approx(0.5)
    assert series_win_prob(0.6, 5) > series_win_prob(0.6, 3) > 0.6
    assert bo3_prob_from_maps(0.6, 0.6, 0.6) == pytest.approx(series_win_prob(0.6, 3))


def test_odds_math():
    assert margin(1.9, 1.9) == pytest.approx(2 / 1.9 - 1)
    p1, p2 = fair_probs(1.9, 1.9)
    assert p1 == pytest.approx(0.5) and p2 == pytest.approx(0.5)
    for method in ("proportional", "power"):
        p1, p2 = fair_probs(1.4, 2.9, method)
        assert p1 + p2 == pytest.approx(1)
        assert 0.65 < p1 < 0.7
    # power сильнее режет андердога, чем proportional
    assert fair_probs(1.4, 2.9, "power")[1] < fair_probs(1.4, 2.9, "proportional")[1]
    assert expected_value(0.6, 1.8) == pytest.approx(0.08)
    assert kelly_fraction(0.6, 1.8, 1.0) == pytest.approx(0.1)
    assert kelly_fraction(0.4, 1.8) == 0


# ---------- названия команд ----------

def test_normalize_and_matcher():
    assert normalize("Team Spirit") == normalize("Spirit") == "spirit"
    assert normalize("Natus Vincere") == "natusvincere"
    m = TeamMatcher(["Natus Vincere", "Spirit", "Spirit Academy", "Virtus.pro", "MOUZ", "The MongolZ"])
    assert m.match("NAVI")[0] == "Natus Vincere"
    assert m.match("Team Spirit")[0] == "Spirit"
    assert m.match("Virtus Pro")[0] == "Virtus.pro"
    assert m.match("mousesports")[0] == "MOUZ"
    assert m.match("MongolZ")[0] == "The MongolZ"
    assert m.match("Совершенно другая команда")[0] is None


# ---------- рейтинги и модель ----------

def _m(i, a, b, s1, s2, bo=3, maps=None, days=0):
    return Match(f"t{i}", T0 + timedelta(days=days, hours=i), a, b, s1, s2, bo, maps=maps or [])


def test_elo_updates_and_veto():
    e = RatingEngine()
    for i in range(20):
        e.update(_m(i, "A", "B", 2, 0, maps=[MapResult("mirage", 1), MapResult("nuke", 1)]))
    assert e.get_rating("A") > 1500 > e.get_rating("B")
    assert e.get_rating("A") + e.get_rating("B") == pytest.approx(3000, abs=1e-6)
    assert e.p_map("A", "B") > 0.8
    assert e.map_off["A"]["mirage"] > 0 > e.map_off["B"]["mirage"]
    # простой долгий — рейтинг откатывается к среднему
    later = T0 + timedelta(days=200)
    assert 1500 < e.get_rating("A", later) < e.get_rating("A")


def test_veto_uses_map_pool():
    e = RatingEngine()
    maps = ["ancient", "anubis", "dust2", "inferno", "mirage", "nuke", "train"]
    for i in range(40):  # A сильна на mirage/nuke, слаба на остальных против B
        m = maps[i % 7]
        e.update(_m(i, "A", "B", 1, 0, 1, [MapResult(m, 1 if m in ("mirage", "nuke") else 2)]))
    assert len(e.active_pool()) == 7
    assert e.veto_bo3("A", "B") is not None


def test_logistic_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    true_w = np.array([1.2, 0.5, 0, 0, 0])
    y = (rng.random(4000) < 1 / (1 + np.exp(-X @ true_w))).astype(float)
    w = LogisticModel(l2=1.0).fit(X, y).w
    assert w[0] == pytest.approx(1.2, abs=0.15) and w[1] == pytest.approx(0.5, abs=0.15)
    assert abs(w[2]) < 0.15


def test_backtest_on_synthetic_beats_coinflip():
    matches = generate(n_teams=40, days=200, matches_per_day=10, seed=3)
    rows = run_backtest(matches, burn_in=300)
    acc = sum((r.p >= 0.5) == bool(r.win1) for r in rows) / len(rows)
    assert acc > 0.55
    text = report(rows, Settings())
    assert "Симуляция ставок" in text and "bets" in text
    assert top_daily(rows, 3)["n"] > 0


def test_backtest_has_no_lookahead():
    """Прогноз матча не должен зависеть от его собственного результата."""
    matches = generate(n_teams=30, days=60, matches_per_day=8, seed=5)
    rows_a = run_backtest(matches, burn_in=100)
    flipped = list(matches)
    last = flipped[-1]
    flipped[-1] = Match(last.match_id, last.date, last.team1, last.team2, last.score2, last.score1, last.best_of)
    rows_b = run_backtest(flipped, burn_in=100)
    assert rows_a[-1].p == pytest.approx(rows_b[-1].p)
    assert rows_a[-1].win1 != rows_b[-1].win1


# ---------- хранение ----------

def test_storage_roundtrip_and_merge(tmp_path):
    a = Match("hltv:1", T0, "Vitality", "Spirit", 2, 1, 3, "Ev", "hltv", [MapResult("mirage", 1)], 1.7, 2.1)
    b = Match("ps:9", T0 + timedelta(hours=2), "Team Vitality", "Team Spirit", 2, 1, 3, "Ev", "pandascore")
    c = Match("hltv:2", T0 + timedelta(hours=5), "Vitality", "Spirit", 0, 2, 3, "Ev", "hltv")
    merged = merge_matches([a], [b, c])
    assert [m.match_id for m in merged] == ["hltv:1", "hltv:2"]  # дубль из PandaScore убран, реванш — нет
    p = tmp_path / "m.csv"
    save_matches(p, merged)
    back = load_matches(p)
    assert back[0].maps[0].name == "mirage" and back[0].odds1 == 1.7 and back[1].winner == 2


# ---------- парсеры ----------

def test_hltv_results_parser():
    ms = parse_results_page((FIX / "hltv_results.html").read_text())
    assert len(ms) == 2
    bo3 = next(m for m in ms if m.best_of == 3)
    assert (bo3.team1, bo3.team2, bo3.score1, bo3.score2) == ("Vitality", "Spirit", 2, 1)
    bo1 = next(m for m in ms if m.best_of == 1)
    assert (bo1.score1, bo1.score2, bo1.winner) == (0, 1, 2)
    assert bo1.maps[0].name == "nuke" and bo1.maps[0].winner == 2
    assert bo3.date.year == 2026


def test_hltv_match_maps_parser():
    maps = parse_match_maps((FIX / "hltv_match.html").read_text(), "Vitality")
    assert [(m.name, m.winner) for m in maps] == [("mirage", 1), ("inferno", 2), ("nuke", 1)]


def test_hltv_upcoming_parser():
    up = parse_upcoming_page((FIX / "hltv_upcoming.html").read_text())  # FURIA-PaiN продублирован — должен остаться один
    assert [(u["team1"], u["team2"], u["best_of"]) for u in up] == [("Natus Vincere", "G2", 3), ("FURIA", "PaiN", 3)]


@pytest.mark.parametrize("payload", [
    # вариант 1: команды полями, рынки списком
    {"events": [{"id": 1, "startTime": 1790100000, "team1": {"name": "NAVI"}, "team2": {"name": "G2"},
                 "format": "Bo3", "markets": [
                     {"name": "Тотал карт", "outcomes": [{"name": "Б 2.5", "coef": 2.1}, {"name": "М 2.5", "coef": 1.7}]},
                     {"name": "Победитель матча", "outcomes": [{"name": "П1", "coef": 1.72}, {"name": "П2", "coef": 2.05}]}]}]},
    # вариант 2: список участников, название "A - B", коэффициенты *1000
    {"data": {"sport": [{"tournamentName": "IEM", "matches": [
        {"eventId": "x", "name": "NAVI - G2", "start": "2026-09-23T18:00:00Z", "competitors": [{"title": "NAVI"}, {"title": "G2"}],
         "odds": [{"type": "W1", "price": 1720}, {"type": "W2", "price": 2050}], "desc": "best of 3"}]}]}},
])
def test_betboom_extractor(payload):
    evs = extract_events(payload)
    assert len(evs) == 1
    ev = evs[0]
    assert (ev.team1, ev.team2, ev.odds1, ev.odds2, ev.best_of) == ("NAVI", "G2", 1.72, 2.05, 3)
    assert ev.start is not None


def test_betboom_manual_csv(tmp_path):
    p = tmp_path / "bb.csv"
    p.write_text("start,team1,team2,odds1,odds2,best_of\n2026-09-30T15:00:00Z,NAVI,G2,\"1,72\",2.05,3\n", encoding="utf-8")
    ev = from_csv(p)[0]
    assert ev.odds1 == 1.72 and ev.best_of == 3


# ---------- сигналы ----------

def _trained():
    hist = []
    for i in range(60):  # Strong стабильно сильнее Mid, Mid сильнее Weak
        hist.append(_m(3 * i, "Strong", "Mid", 2, 1 if i % 3 else 0, days=i))
        hist.append(_m(3 * i + 1, "Mid", "Weak", 2, 1, days=i))
        hist.append(_m(3 * i + 2, "Strong", "Weak", 2, 0, days=i))
    return Predictor(burn_in=30).train(hist), hist


def test_signals_filter():
    pred, hist = _trained()
    now = hist[-1].date + timedelta(hours=1)
    matcher = TeamMatcher(sorted(pred.engine.rating))
    line = [
        OddsEvent("Team Strong", "Mid", 1.80, 2.00, now + timedelta(hours=5), 3, "Cup"),   # фаворит модели по хорошему кэфу
        OddsEvent("Strong", "Mid", 1.80, 2.00, now + timedelta(hours=5), 1, "Cup"),        # Bo1 — отсекаем
        OddsEvent("Strong", "Weak", 1.10, 7.00, now + timedelta(hours=5), 3, "Cup"),       # кэф вне диапазона
        OddsEvent("Unknown", "Mid", 1.70, 2.10, now + timedelta(hours=5), 3, "Cup"),       # нет истории
        OddsEvent("Strong", "Mid", 1.80, 2.00, now + timedelta(hours=80), 3, "Cup"),       # за горизонтом
        OddsEvent("Mid", "Weak", 1.75, 2.05, now + timedelta(hours=5), None, "Cup"),       # формат из расписания
    ]
    schedule = [{"team1": "Weak", "team2": "Mid", "best_of": 3}]
    s = Settings()
    s.max_ev = 5.0
    rows = evaluate(line, pred, matcher, schedule, s, now=now)
    assert len(rows) == 5
    by = {(r.team1, r.team2, r.best_of): r for r in rows}
    sig = by[("Team Strong", "Mid", 3)]
    assert sig.signal and sig.pick == "Team Strong" and sig.pick_ev >= 0.07
    assert not by[("Strong", "Mid", 1)].signal and "Bo1" in by[("Strong", "Mid", 1)].reason
    assert "кэф вне диапазона" in by[("Strong", "Weak", 3)].reason
    assert "не найдена" in by[("Unknown", "Mid", 3)].reason
    assert by[("Mid", "Weak", 3)].best_of == 3
    top = most_probable(rows, 3)
    assert top[0].pick == "Strong" and top[0].pick_p > 0.8


def test_odds_log_attach(tmp_path):
    pred, hist = _trained()
    p = tmp_path / "log.csv"
    target = hist[-1]
    log_odds([OddsEvent(target.team2, target.team1, 2.4, 1.5, target.date)], p, now=target.date - timedelta(hours=3))
    matcher = TeamMatcher(sorted(pred.engine.rating))
    assert attach_odds_log(hist, p, matcher) == 1
    assert (target.odds1, target.odds2) == (1.5, 2.4)  # ориентация по team1/team2 истории


def test_betboom_text_extractor():
    """Текст страницы линии (формат примерный — реальная вёрстка BetBoom может отличаться)."""
    from cs_predictor.sources.betboom import extract_from_text
    text = """Линия
Кибер
Counter-Strike
ESL Pro League
Сегодня
19:30
Bo3
Natus Vincere
G2
П1
1,72
П2
2,05
+45
30 сентября
12:00
Spirit
FaZe
1.55
2.40
Футбол-матч
Команда А
Команда Б
2.10
3.30
3.40
"""
    now = datetime(2026, 9, 29, 9, 0, tzinfo=timezone.utc)
    evs = {(e.team1, e.team2): e for e in extract_from_text(text, now)}
    assert set(evs) == {("Natus Vincere", "G2"), ("Spirit", "FaZe")}  # 3-исходный рынок пропущен
    navi = evs[("Natus Vincere", "G2")]
    assert (navi.odds1, navi.odds2, navi.best_of) == (1.72, 2.05, 3)
    assert navi.start == datetime(2026, 9, 29, 16, 30, tzinfo=timezone.utc)  # 19:30 МСК
    assert evs[("Spirit", "FaZe")].start == datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
