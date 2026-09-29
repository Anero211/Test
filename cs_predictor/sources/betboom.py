"""Линия BetBoom по CS2: матчи и коэффициенты П1/П2.

Публичного API у BetBoom нет, поэтому есть три режима:

1. playwright — headless-браузер открывает страницу CS2 и перехватывает все JSON-ответы,
   из которых универсальный экстрактор достаёт матчи и коэффициенты;
2. url — прямой JSON-эндпоинт, скопированный из DevTools → Network → Fetch/XHR
   (переменная BETBOOM_JSON_URL);
3. csv — ручной файл data/betboom_manual.csv (start,team1,team2,odds1,odds2,best_of).

Если экстрактор находит 0 матчей — запусти `python -m cs_predictor betboom-dump`: все
перехваченные JSON сохранятся в data/betboom_dump/, по ним легко дописать точный парсер.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..storage import parse_dt

log = logging.getLogger(__name__)


@dataclass
class OddsEvent:
    team1: str
    team2: str
    odds1: float
    odds2: float
    start: datetime | None = None
    best_of: int | None = None
    tournament: str = ""
    event_id: str = ""
    source: str = "betboom"


# ---------------- универсальный экстрактор из произвольного JSON ----------------

_NAME_KEYS = ("name", "title", "shortName", "short_name", "caption", "label", "nameRu", "name_ru", "value")
_PAIR_KEYS = [("team1", "team2"), ("home", "away"), ("homeTeam", "awayTeam"), ("home_team", "away_team"),
              ("competitor1", "competitor2"), ("participant1", "participant2"), ("opp1", "opp2"),
              ("player1", "player2"), ("team1Name", "team2Name")]
_LIST_KEYS = ("competitors", "participants", "teams", "opponents", "players", "sides")
_ODD_KEYS = ("odds", "odd", "coef", "coefficient", "price", "k", "rate", "factor", "koef", "value", "v")
_START_KEYS = ("startTime", "start_time", "startDate", "start_date", "startAt", "start_at", "begin_at",
               "beginAt", "date", "time", "start", "kickoff", "scheduled", "eventDate", "startTs", "ts")
_WINNER_MARKET = re.compile(r"(winner|победител|исход|moneyline|match\s*result|1x2|^1-2$|^12$|итог)", re.I)
_BAD_MARKET = re.compile(r"(карт|map|тотал|total|фора|handicap|раунд|round|пистол|pistol|тайм|half)", re.I)
_BO_RE = re.compile(r"\b(?:bo|best\s*of)\s*-?(\d)\b", re.I)
_SPLIT_RE = re.compile(r"\s+(?:-|–|—|vs\.?|v)\s+", re.I)


def _name_of(x: Any) -> str | None:
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, dict):
        for k in _NAME_KEYS:
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                n = _name_of(v)
                if n:
                    return n
        for k in ("team", "competitor", "participant", "opponent"):
            if isinstance(x.get(k), dict):
                return _name_of(x[k])
    return None


def _teams(d: dict) -> tuple[str, str] | None:
    for k1, k2 in _PAIR_KEYS:
        if k1 in d and k2 in d:
            a, b = _name_of(d[k1]), _name_of(d[k2])
            if a and b:
                return a, b
    for k in _LIST_KEYS:
        v = d.get(k)
        if isinstance(v, list) and len(v) == 2:
            a, b = _name_of(v[0]), _name_of(v[1])
            if a and b:
                return a, b
    for k in ("name", "title", "eventName", "event_name", "matchName"):
        v = d.get(k)
        if isinstance(v, str):
            parts = _SPLIT_RE.split(v.strip())
            if len(parts) == 2 and all(parts):
                return parts[0].strip(), parts[1].strip()
    return None


def _as_odd(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if 1.005 < f < 100:
        return f
    if 1005 < f < 100000:  # иногда коэффициенты приходят как int*1000
        return f / 1000
    return None


def _outcome_odd(o: dict) -> float | None:
    for k in _ODD_KEYS:
        if k in o and not isinstance(o[k], (dict, list)):
            f = _as_odd(o[k])
            if f:
                return f
    return None


def _markets(d: dict, depth: int = 0) -> Iterable[tuple[str, list[dict]]]:
    """Все списки из >=2 словарей с коэффициентами внутри события (= рынки/исходы)."""
    if depth > 5:
        return
    for k, v in d.items():
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, dict) for x in v):
            if sum(1 for x in v if _outcome_odd(x)) >= 2:
                label = str(d.get("name") or d.get("title") or d.get("marketName") or d.get("type") or k)
                yield label, v
            for x in v:
                yield from _markets(x, depth + 1)
        elif isinstance(v, dict):
            yield from _markets(v, depth + 1)


def _pick_winner_odds(d: dict, t1: str, t2: str) -> tuple[float, float] | None:
    candidates = []
    for label, outcomes in _markets(d):
        odds = [(o, _outcome_odd(o)) for o in outcomes if _outcome_odd(o)]
        if len(odds) != 2:  # в CS2 ничьей в серии нет (кроме Bo2) — рынок П1/П2 ровно из 2 исходов
            continue
        score = 0
        if _WINNER_MARKET.search(label):
            score += 3
        if _BAD_MARKET.search(label):
            score -= 5
        names = [(_name_of(o) or "").lower() for o, _ in odds]
        if names[0] in ("1", "п1", "w1", t1.lower()) and names[1] in ("2", "п2", "w2", t2.lower()):
            score += 2
        if names[0] in ("2", "п2", "w2", t2.lower()):  # перепутан порядок
            odds.reverse()
            score += 2
        candidates.append((score, odds[0][1], odds[1][1]))
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c[0])
    return best[1], best[2]


def _start(d: dict) -> datetime | None:
    for k in _START_KEYS:
        if k in d and not isinstance(d[k], (dict, list)):
            dt = parse_dt(d[k])
            if dt and dt.year >= 2020:
                return dt
    return None


def _best_of(d: dict) -> int | None:
    for k in ("bestOf", "best_of", "bo", "format", "numberOfGames", "number_of_games", "mapsCount"):
        v = d.get(k)
        if isinstance(v, int) and 1 <= v <= 5:
            return v
        if isinstance(v, str):
            m = _BO_RE.search(v) or re.fullmatch(r"\s*([1-5])\s*", v)
            if m:
                return int(m.group(1))
    m = _BO_RE.search(json.dumps(d, ensure_ascii=False)[:4000])
    return int(m.group(1)) if m else None


def extract_events(payload: Any) -> list[OddsEvent]:
    """Рекурсивно ищет в JSON объекты «матч»: две команды + рынок из 2 исходов с коэффициентами."""
    found: dict[tuple, OddsEvent] = {}

    def walk(x: Any, depth: int = 0, tournament: str = "") -> None:
        if depth > 30:
            return
        if isinstance(x, dict):
            teams = _teams(x)
            if teams:
                odds = _pick_winner_odds(x, *teams)
                if odds:
                    ev = OddsEvent(teams[0], teams[1], odds[0], odds[1], _start(x), _best_of(x),
                                   tournament, str(x.get("id") or x.get("eventId") or ""))
                    found[(ev.team1.lower(), ev.team2.lower())] = ev
                    return
            t = x.get("tournamentName") or x.get("leagueName") or x.get("champName") or tournament
            for v in x.values():
                walk(v, depth + 1, t if isinstance(t, str) else tournament)
        elif isinstance(x, list):
            for v in x:
                walk(v, depth + 1, tournament)

    walk(payload)
    return list(found.values())


# ---------------- режимы получения ----------------

def from_json_url(url: str) -> list[OddsEvent]:
    from .http import Fetcher
    return extract_events(Fetcher(min_interval=1.0).get_json(url))


def capture_with_playwright(page_url: str, wait_seconds: float = 12.0, dump_dir: Path | None = None) -> list[Any]:
    """Открывает страницу в headless Chromium и собирает все JSON-ответы сайта."""
    from playwright.sync_api import sync_playwright

    payloads: list[Any] = []

    def on_response(resp):
        try:
            if "json" in (resp.headers.get("content-type") or ""):
                payloads.append(resp.json())
                if dump_dir:
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    name = re.sub(r"[^\w.-]+", "_", resp.url.split("//", 1)[-1])[:150]
                    (dump_dir / f"{len(payloads):03d}_{name}.json").write_text(
                        json.dumps(payloads[-1], ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception:
            pass

    import os
    launch: dict[str, Any] = {"headless": True}
    if os.environ.get("HTTPS_PROXY"):  # корпоративный/облачный прокси
        launch["proxy"] = {"server": os.environ["HTTPS_PROXY"]}
    if os.environ.get("BROWSER_EXTRA_ARGS"):  # напр. доверие CA прокси: --ignore-certificate-errors-spki-list=<hash>
        launch["args"] = os.environ["BROWSER_EXTRA_ARGS"].split()
    if os.environ.get("CHROMIUM_EXECUTABLE"):
        launch["executable_path"] = os.environ["CHROMIUM_EXECUTABLE"]
    with sync_playwright() as p:
        browser = p.chromium.launch(**launch)
        page = browser.new_page(locale="ru-RU", viewport={"width": 1400, "height": 1000})
        page.on("response", on_response)
        page.goto(page_url, wait_until="domcontentloaded", timeout=60000)
        deadline = wait_seconds * 1000
        step = 1500
        waited = 0
        while waited < deadline:  # прокручиваем, чтобы подгрузилась вся линия
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(step)
            waited += step
        browser.close()
    return payloads


def from_playwright(page_url: str, dump_dir: Path | None = None) -> list[OddsEvent]:
    events: dict[tuple, OddsEvent] = {}
    for payload in capture_with_playwright(page_url, dump_dir=dump_dir):
        for ev in extract_events(payload):
            events[(ev.team1.lower(), ev.team2.lower())] = ev
    return list(events.values())


def from_csv(path: Path) -> list[OddsEvent]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append(OddsEvent(r["team1"], r["team2"], float(r["odds1"].replace(",", ".")),
                                     float(r["odds2"].replace(",", ".")), parse_dt(r.get("start")),
                                     int(r["best_of"]) if r.get("best_of") else None, r.get("tournament", "")))
            except (KeyError, ValueError) as e:
                log.warning("строка %s пропущена: %s", r, e)
    return out


def fetch_line(mode: str, page_url: str, json_url: str, manual_csv: Path) -> list[OddsEvent]:
    """mode: auto | playwright | url | csv. auto = url (если задан) → playwright → csv."""
    order = {"auto": ["url", "playwright", "csv"]}.get(mode, [mode])
    for m in order:
        try:
            if m == "url" and json_url:
                events = from_json_url(json_url)
            elif m == "playwright":
                events = from_playwright(page_url)
            elif m == "csv":
                events = from_csv(manual_csv)
            else:
                continue
        except Exception as e:
            log.warning("BetBoom [%s] не сработал: %s", m, e)
            continue
        log.info("BetBoom [%s]: %s матчей в линии", m, len(events))
        if events:
            return events
    return []
