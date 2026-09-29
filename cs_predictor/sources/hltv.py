"""Парсер HLTV: результаты матчей (история), карты матча, ближайшие матчи (формат Bo1/Bo3).

HLTV защищён Cloudflare: ставь curl_cffi (`pip install curl_cffi`), он имитирует Chrome.
Вёрстка HLTV иногда меняется — если парсер вернёт 0 матчей, проверь селекторы ниже.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from ..storage import MapResult, Match, parse_dt
from .http import Fetcher

log = logging.getLogger(__name__)
BASE = "https://www.hltv.org"
_BO_RE = re.compile(r"\bbo(\d)\b", re.I)


# Сокращения карт на странице результатов HLTV -> полные названия (как на странице матча)
MAP_ABBR = {"d2": "dust2", "mrg": "mirage", "inf": "inferno", "nuke": "nuke", "anc": "ancient",
            "anb": "anubis", "trn": "train", "ovp": "overpass", "vtg": "vertigo", "cch": "cache",
            "cbl": "cobblestone", "tsc": "tuscan", "de_dust2": "dust2"}


def canonical_map(name: str) -> str:
    n = name.strip().lower().replace(" ", "")
    return MAP_ABBR.get(n, n)


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


def parse_results_page(html: str) -> list[Match]:
    """Разбирает страницу /results. Для Bo1 HLTV показывает счёт раундов и название карты."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, Match] = {}
    for con in soup.select("div.result-con"):
        a = con.select_one("a[href*='/matches/']")
        if not a:
            continue
        mid = re.search(r"/matches/(\d+)/", a["href"])
        ts = con.get("data-zonedgrouping-entry-unix")
        t1 = _text(con.select_one(".team1 .team"))
        t2 = _text(con.select_one(".team2 .team"))
        scores = [s for s in con.select(".result-score span")]
        if not (mid and ts and t1 and t2 and len(scores) >= 2):
            continue
        try:
            s1, s2 = int(_text(scores[0])), int(_text(scores[1]))
        except ValueError:
            continue
        map_text = _text(con.select_one(".map-text")).lower()
        if map_text == "def":  # техническое поражение — не игра
            continue
        bo = _BO_RE.search(map_text)
        maps: list[MapResult] = []
        if bo:
            best_of = int(bo.group(1))
        else:  # Bo1: в map-text название карты, в счёте — раунды
            best_of = 1
            winner = 1 if s1 > s2 else 2
            if map_text and map_text != "-":
                maps = [MapResult(canonical_map(map_text), winner)]
            s1, s2 = int(winner == 1), int(winner == 2)
        out[mid.group(1)] = Match(
            match_id=f"hltv:{mid.group(1)}", date=parse_dt(ts), team1=t1, team2=t2,
            score1=s1, score2=s2, best_of=best_of, event=_text(con.select_one(".event-name")),
            source="hltv", maps=maps,
        )
        out[mid.group(1)].url = BASE + a["href"]  # type: ignore[attr-defined]
    return list(out.values())


def parse_match_maps(html: str, team1: str) -> list[MapResult]:
    """Карты и победители со страницы матча /matches/<id>/..."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for holder in soup.select("div.mapholder"):
        name = canonical_map(_text(holder.select_one(".mapname")))
        left = holder.select_one(".results-left")
        right = holder.select_one(".results-right")
        if not name or name == "tba" or not left or not right:
            continue
        left_team = _text(left.select_one(".results-teamname"))
        lc, rc = " ".join(left.get("class", [])), " ".join(right.get("class", []))
        if "won" in lc:
            left_won = True
        elif "won" in rc:
            left_won = False
        else:
            try:
                ls, rs = int(_text(left.select_one(".results-team-score"))), int(_text(right.select_one(".results-team-score")))
            except ValueError:
                continue  # карта не сыграна
            if ls == rs:
                continue
            left_won = ls > rs
        left_is_t1 = not left_team or left_team.lower() == team1.lower()
        out.append(MapResult(name, 1 if left_won == left_is_t1 else 2))
    return out


def parse_upcoming_page(html: str) -> list[dict]:
    """Ближайшие матчи /matches. Поддерживает старую (upcomingMatch) и новую (match-wrapper) вёрстку."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[tuple, dict] = {}
    for el in soup.select("div.upcomingMatch, div.match-wrapper"):
        names = [_text(x) for x in el.select(".matchTeamName, .match-teamname, .team-name")]
        names = [n for n in names if n and n.upper() != "TBD"]
        ts = (el.get("data-zonedgrouping-entry-unix")
              or (el.select_one("[data-unix]") or {}).get("data-unix"))
        meta = _text(el.select_one(".matchMeta, .match-meta"))
        bo = _BO_RE.search(meta)
        if len(names) >= 2 and ts:
            best_of = int(bo.group(1)) if bo else None
            if best_of is None and meta.strip().lower() in set(MAP_ABBR) | set(MAP_ABBR.values()):
                best_of = 1  # для Bo1 с известной картой HLTV пишет карту вместо «bo1»
            item = {"team1": names[0], "team2": names[1], "start": parse_dt(ts), "best_of": best_of,
                    "event": _text(el.select_one(".matchEventName, .match-event")), "source": "hltv"}
            key = (item["team1"], item["team2"], item["start"])
            # в новой вёрстке блоки вложены друг в друга — оставляем самый полный
            if key not in out:
                out[key] = item
            else:  # объединяем: у одного блока может быть формат, у другого — турнир
                for k in ("best_of", "event"):
                    out[key][k] = out[key][k] or item[k]
    return list(out.values())


class HLTVClient:
    def __init__(self, fetcher: Fetcher | None = None):
        self.f = fetcher or Fetcher(min_interval=3.0)

    def results(self, pages: int = 10, days: int | None = None, with_maps: bool = False) -> list[Match]:
        """История. 1 страница = 100 матчей. with_maps=True дополнительно открывает страницы Bo3
        (медленно: +1 запрос на матч), зато модель узнаёт маппул команд."""
        matches: list[Match] = []
        stop = datetime.now(timezone.utc) - timedelta(days=days) if days else None
        for page in range(pages):
            try:
                html = self.f.get_text(f"{BASE}/results", params={"offset": page * 100})
            except Exception as e:  # не теряем уже скачанное
                if not matches:
                    raise
                log.warning("HLTV остановлен на offset=%s: %s", page * 100, e)
                break
            batch = parse_results_page(html)
            log.info("HLTV results offset=%s: %s матчей", page * 100, len(batch))
            if not batch:
                break
            matches += batch
            if stop and min(m.date for m in batch) < stop:
                break
        if stop:
            matches = [m for m in matches if m.date >= stop]
        if with_maps:
            for m in matches:
                if m.best_of > 1 and not m.maps:
                    try:
                        m.maps = parse_match_maps(self.f.get_text(m.url), m.team1)  # type: ignore[attr-defined]
                    except Exception as e:
                        log.warning("карты %s: %s", m.match_id, e)
        return matches

    def upcoming(self, cache: "Path | None" = None, max_age_hours: float = 12) -> list[dict]:
        """Расписание. Удачный ответ кэшируется; если HLTV временно блокирует (Cloudflare),
        используется кэш не старше max_age_hours."""
        import time as _t
        try:
            html = self.f.get_text(f"{BASE}/matches")
        except Exception as e:
            if cache and cache.exists() and _t.time() - cache.stat().st_mtime < max_age_hours * 3600:
                age = (_t.time() - cache.stat().st_mtime) / 3600
                log.warning("HLTV недоступен (%s) — беру расписание из кэша (%.1f ч назад)", e, age)
                return parse_upcoming_page(cache.read_text(encoding="utf-8"))
            raise
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(html, encoding="utf-8")
        return parse_upcoming_page(html)
