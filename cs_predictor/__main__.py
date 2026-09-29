"""CLI: python -m cs_predictor <команда>

  collect       — скачать историю матчей (HLTV и/или PandaScore) в data/matches.csv
  backtest      — проверить модель на прошлых матчах (walk-forward)
  tune          — подобрать параметры Elo на истории (сохраняются в data/params.json)
  predict       — предстоящие матчи HLTV: вероятности, счёт по картам, с какого кэфа ставить
  watch         — predict в цикле + новые матчи в Telegram
  betboom       — сверка прогнозов с линией BetBoom (EV, сигналы) — только на своём компьютере
  betboom-dump  — сохранить все JSON-ответы сайта BetBoom (для настройки парсера)
  demo          — всё то же на синтетических данных (проверка установки без интернета)
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from . import backtest as bt
from .config import DATA_DIR, OUTPUT_DIR, Settings
from .model import Predictor
from .names import TeamMatcher
from .strategy import MatchStrategy, build_strategy, fmt_time, match_card, summary_table, telegram_text
from .signals import evaluate, format_table, log_odds, most_probable, save_rows
from .sources import betboom
from .storage import load_matches, merge_matches, parse_dt, save_matches

log = logging.getLogger("cs_predictor")
PARAMS = DATA_DIR / "params.json"


def _date(s: str | None) -> datetime | None:
    return parse_dt(s) if s else None


def cmd_collect(a, s: Settings) -> None:
    new = []
    if a.source in ("hltv", "all"):
        from .sources.hltv import HLTVClient
        try:
            new += HLTVClient().results(pages=a.pages, days=a.days, with_maps=a.maps)
        except Exception as e:
            log.error("HLTV недоступен: %s", e)
    if a.source in ("pandascore", "all"):
        if s.pandascore_token:
            from .sources.pandascore import PandaScoreClient
            try:
                new += PandaScoreClient(s.pandascore_token).past(pages=min(a.pages, 30), days=a.days)  # бережём лимит 1000 запр/мес
            except Exception as e:
                log.error("PandaScore недоступен: %s", e)
        else:
            log.warning("PANDASCORE_TOKEN не задан — PandaScore пропущен")
    merged = merge_matches(load_matches(s.matches_csv), new)
    save_matches(s.matches_csv, merged)
    print(f"Скачано {len(new)} матчей, в базе {len(merged)} → {s.matches_csv}")


def _print_backtest(rows, s: Settings, title: str) -> None:
    text = bt.report(rows, s, title)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "backtest_report.md").write_text(text, encoding="utf-8")
    bt.save_rows(rows, OUTPUT_DIR / "backtest_predictions.csv")
    print(text)
    print(f"Отчёт: {OUTPUT_DIR / 'backtest_report.md'}, прогнозы по матчам: {OUTPUT_DIR / 'backtest_predictions.csv'}")


def cmd_backtest(a, s: Settings) -> None:
    matches = load_matches(s.matches_csv)
    if not matches:
        sys.exit("История пуста — сначала `python -m cs_predictor collect`")
    if a.odds_log:
        matcher = TeamMatcher(sorted({t for m in matches for t in (m.team1, m.team2)}), s.aliases_json)
        print(f"Коэффициенты BetBoom найдены для {bt.attach_odds_log(matches, s.odds_log_csv, matcher)} матчей")
    rows = bt.run_backtest(matches, _date(a.since), _date(a.until), burn_in=a.burn_in, params=bt.load_params(PARAMS))
    _print_backtest(rows, s, f"Бэктест на реальной истории ({len(matches)} матчей в базе)")


def cmd_tune(a, s: Settings) -> None:
    matches = load_matches(s.matches_csv)
    if not matches:
        sys.exit("История пуста — сначала `python -m cs_predictor collect`")
    if a.until:  # подбираем только на прошлом, чтобы бэктест после этой даты был честным
        matches = [m for m in matches if m.date < _date(a.until)]
    best, results = bt.tune(matches, s)
    print(f"{'K':>4} {'K_map':>6} {'shrink':>7} {'N':>6} {'acc':>7} {'logloss':>8}")
    for r in results:
        print(f"{r['k']:>4} {r['k_map']:>6} {r['inactivity_shrink']:>7} {r['n']:>6} {r['acc']:>7.1%} {r['logloss']:>8.4f}")
    bt.save_params(best, PARAMS)
    print(f"Лучшие параметры сохранены в {PARAMS}")


def _schedule(s: Settings, hours: float) -> list[dict]:
    """Расписание (для определения Bo1/Bo3, если BetBoom его не отдаёт)."""
    out = []
    if s.pandascore_token:
        from .sources.pandascore import PandaScoreClient
        try:
            out += PandaScoreClient(s.pandascore_token).upcoming(hours)
        except Exception as e:
            log.warning("PandaScore расписание: %s", e)
    try:
        from .sources.hltv import HLTVClient
        out += HLTVClient().upcoming()
    except Exception as e:
        log.warning("HLTV расписание: %s", e)
    return out


def _load_predictor(s: Settings, refresh: bool) -> Predictor:
    if refresh:  # подтягиваем свежие результаты (последние ~3 дня)
        cmd_collect(argparse.Namespace(source="all", pages=3, days=3, maps=False), s)
    matches = load_matches(s.matches_csv)
    if not matches:
        sys.exit("История пуста — сначала `python -m cs_predictor collect`")
    return Predictor(bt.load_params(PARAMS)).train(matches)


# ---------------- основной режим: матчи HLTV → стратегия ----------------

def run_hltv_predict(s: Settings, hours: float, top: int, formats: tuple[int, ...], refresh: bool,
                     margin: float, quiet: bool = False) -> list[MatchStrategy]:
    predictor = _load_predictor(s, refresh)
    matcher = TeamMatcher(sorted(predictor.engine.rating), s.aliases_json)
    now = datetime.now(timezone.utc)
    items: list[MatchStrategy] = []
    skipped = []
    for u in _schedule(s, hours):
        start, bo = u.get("start"), u.get("best_of") or 3
        if not start or not (now - timedelta(minutes=15) <= start <= now + timedelta(hours=hours)):
            continue
        if bo not in formats:
            continue
        t1, t2 = matcher.match(u["team1"])[0], matcher.match(u["team2"])[0]
        if not (t1 and t2) or t1 == t2:
            skipped.append(f"{u['team1']} – {u['team2']} (команда не найдена в истории)")
            continue
        pr = predictor.predict(t1, t2, bo, start)
        if min(pr.games1, pr.games2) < s.min_team_games:
            skipped.append(f"{u['team1']} – {u['team2']} (мало матчей в истории)")
            continue
        items.append(build_strategy(pr, start, u.get("event", ""), u["team1"], u["team2"], margin=margin))
    # дубли из разных источников расписания
    uniq = {}
    for st in items:
        uniq.setdefault((frozenset((st.team1.lower(), st.team2.lower())), st.start.date() if st.start else None), st)
    items = sorted(uniq.values(), key=lambda st: -st.fav_p)

    stamp = datetime.now(timezone.utc)
    parts = [f"=== Прогнозы CS2 на {hours:.0f} ч по расписанию HLTV — {fmt_time(stamp)} МСК ===",
             f"«Ставить от» = минимальный коэффициент, при котором ставка выгодна с запасом {margin:.0%} "
             f"(+5% для команд с короткой историей). Ниже этого кэфа — пропускать.", "",
             summary_table(items), ""]
    parts += [match_card(st, i) + "\n" for i, st in enumerate(items[:top], 1)]
    if skipped:
        parts.append(f"Пропущено {len(skipped)} матчей: " + "; ".join(skipped[:15]) + (" …" if len(skipped) > 15 else ""))
    text = "\n".join(parts)
    OUTPUT_DIR.mkdir(exist_ok=True)
    name = f"strategy_{stamp:%Y%m%d_%H%M}"
    (OUTPUT_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
    _save_markets(items, OUTPUT_DIR / f"{name}.csv")
    if not quiet:
        print(text)
        print(f"\nСохранено: {OUTPUT_DIR / (name + '.txt')} и .csv (все рынки)")
    return items


def _save_markets(items: list[MatchStrategy], path) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_msk", "team1", "team2", "best_of", "event", "market", "probability",
                    "fair_odds", "min_odds", "risk", "games1", "games2"])
        for st in items:
            for mk in st.markets:
                w.writerow([fmt_time(st.start), st.team1, st.team2, st.best_of, st.event, mk.label,
                            f"{mk.p:.4f}", f"{mk.fair_odds:.3f}", f"{mk.min_odds:.3f}", mk.risk, st.games1, st.games2])


def _formats(a) -> tuple[int, ...]:
    return (1, 3, 5) if a.bo1 else (3, 5)


def cmd_predict(a, s: Settings) -> None:
    items = run_hltv_predict(s, a.hours, a.top, _formats(a), not a.no_update, a.margin)
    if a.telegram:
        _send_telegram(items[:a.top], s)


def _send_telegram(items: list[MatchStrategy], s: Settings) -> None:
    import json
    from .notify import send_telegram
    if not (s.telegram_token and s.telegram_chat_id):
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы")
        return
    sent = json.loads(s.sent_json.read_text()) if s.sent_json.exists() else {}
    n = 0
    for st in items:
        key = f"{fmt_time(st.start)}|{st.team1}|{st.team2}"
        if key not in sent and send_telegram(s.telegram_token, s.telegram_chat_id, telegram_text(st)):
            sent[key] = round(st.fav_p, 3)
            n += 1
    s.sent_json.parent.mkdir(parents=True, exist_ok=True)
    s.sent_json.write_text(json.dumps(sent, ensure_ascii=False, indent=1))
    print(f"В Telegram отправлено: {n}")


def cmd_watch(a, s: Settings) -> None:
    """Каждые N минут: обновить историю и расписание, прислать в Telegram новые матчи."""
    while True:
        try:
            items = run_hltv_predict(s, a.hours, a.top, _formats(a), True, a.margin, quiet=True)
            _send_telegram(items[:a.top], s)
        except SystemExit as e:
            log.error("%s", e)
        except Exception:
            log.exception("ошибка цикла")
        time.sleep(a.interval + random.uniform(0, 30))


# ---------------- режим с линией BetBoom (запускать на своём компьютере) ----------------

def cmd_betboom(a, s: Settings) -> None:
    if a.hours:
        s.horizon_hours = a.hours
    predictor = _load_predictor(s, not a.no_update)
    matcher = TeamMatcher(sorted(predictor.engine.rating), s.aliases_json)
    line = betboom.fetch_line(a.mode, s.betboom_page_url, s.betboom_json_url, s.betboom_manual_csv)
    if not line:
        sys.exit("Линия BetBoom пуста. Варианты: BETBOOM_JSON_URL из DevTools, `betboom-dump`, "
                 "или заполни data/betboom_manual.csv")
    log_odds(line, s.odds_log_csv)
    need_schedule = any(ev.best_of is None for ev in line)
    rows = evaluate(line, predictor, matcher, _schedule(s, s.horizon_hours) if need_schedule else [], s)
    _output(rows, s, a.top, a.telegram)


def _output(rows, s: Settings, top: int, telegram: bool) -> None:
    signals = [r for r in rows if r.signal]
    print(format_table(signals, f"=== СИГНАЛЫ (Bo3, кэф {s.min_odds}–{s.max_odds}, EV ≥ {s.min_ev:.0%}) ==="))
    print(format_table(most_probable(rows, top), f"=== ТОП-{top} САМЫХ ВЕРОЯТНЫХ ИСХОДОВ (Bo3) на {s.horizon_hours:.0f} ч ==="))
    print(format_table(rows, "=== ВСЯ ЛИНИЯ BetBoom ==="))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    save_rows(rows, OUTPUT_DIR / f"predictions_{stamp}.csv")
    print(f"Сохранено: {OUTPUT_DIR / f'predictions_{stamp}.csv'}")
    if telegram:
        if not (s.telegram_token and s.telegram_chat_id):
            log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы")
        else:
            from .notify import send_new_signals
            print(f"В Telegram отправлено сигналов: {send_new_signals(rows, s.telegram_token, s.telegram_chat_id, s.sent_json)}")


def cmd_betboom_dump(a, s: Settings) -> None:
    d = DATA_DIR / "betboom_dump"
    payloads, text = betboom.capture_with_playwright(s.betboom_page_url, wait_seconds=a.wait, dump_dir=d)
    events = betboom.extract_from_text(text) + [ev for p in payloads for ev in betboom.extract_events(p)]
    print(f"JSON-ответов: {len(payloads)}, строк текста: {text.count(chr(10))} → {d} (page.html, page.txt, page.png)")
    print(f"Распознано матчей: {len(events)}")
    for ev in events[:40]:
        print(f"  {ev.start} {ev.team1} vs {ev.team2}: {ev.odds1} / {ev.odds2} Bo{ev.best_of or '?'}")


def cmd_demo(a, s: Settings) -> None:
    """Синтетика: проверка, что весь конвейер работает без интернета."""
    from .synthetic import generate
    matches = generate(seed=a.seed)
    split = int(len(matches) * 0.97)
    history, future = matches[:split], matches[split:]
    rows = bt.run_backtest(history)
    _print_backtest(rows, s, "ДЕМО-бэктест на СИНТЕТИЧЕСКИХ данных (не отражает реальную точность)")

    predictor = Predictor().train(history)
    matcher = TeamMatcher(sorted(predictor.engine.rating), s.aliases_json)
    now = future[0].date - timedelta(hours=1)
    line = [betboom.OddsEvent(m.team1, m.team2, m.odds1, m.odds2, m.date, m.best_of, "Synthetic Cup")
            for m in future if m.date <= now + timedelta(hours=s.horizon_hours)]
    _output(evaluate(line, predictor, matcher, [], s, now=now), s, a.top, telegram=False)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    s = Settings.from_env()
    p = argparse.ArgumentParser(prog="cs_predictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="скачать историю матчей")
    c.add_argument("--source", choices=["hltv", "pandascore", "all"], default="all")
    c.add_argument("--pages", type=int, default=200, help="максимум страниц по 100 матчей")
    c.add_argument("--days", type=int, default=540, help="глубина истории, дней")
    c.add_argument("--maps", action="store_true", help="HLTV: скачать карты Bo3 (медленно, но точнее маппул)")
    c.set_defaults(func=cmd_collect)

    b = sub.add_parser("backtest", help="проверка на прошлых матчах")
    b.add_argument("--since", help="оценивать матчи начиная с даты YYYY-MM-DD")
    b.add_argument("--until", help="... и до даты")
    b.add_argument("--burn-in", type=int, default=500, help="сколько первых матчей только для разогрева рейтингов")
    b.add_argument("--odds-log", action="store_true", help="подставить накопленные коэффициенты BetBoom и посчитать ROI")
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser("tune", help="подбор параметров Elo")
    t.add_argument("--until", help="использовать для подбора только матчи до этой даты")
    t.set_defaults(func=cmd_tune)

    for name, fn in (("predict", cmd_predict), ("watch", cmd_watch)):
        x = sub.add_parser(name, help="матчи HLTV → вероятности, счёт по картам, с какого кэфа ставить")
        x.add_argument("--hours", type=float, default=48, help="горизонт, часов")
        x.add_argument("--top", type=int, default=15, help="сколько подробных карточек выводить")
        x.add_argument("--bo1", action="store_true", help="включить Bo1 (по умолчанию только Bo3/Bo5)")
        x.add_argument("--margin", type=float, default=0.07, help="запас прочности: минимальный EV (0.07 = 7%%)")
        if name == "predict":
            x.add_argument("--telegram", action="store_true")
            x.add_argument("--no-update", action="store_true", help="не обновлять историю перед прогнозом")
        else:
            x.add_argument("--interval", type=int, default=1800, help="секунд между обновлениями")
        x.set_defaults(func=fn)

    bb = sub.add_parser("betboom", help="сверка с линией BetBoom (только на своём компьютере)")
    bb.add_argument("--mode", choices=["auto", "playwright", "url", "csv"], default="auto")
    bb.add_argument("--top", type=int, default=10)
    bb.add_argument("--hours", type=float)
    bb.add_argument("--telegram", action="store_true")
    bb.add_argument("--no-update", action="store_true")
    bb.set_defaults(func=cmd_betboom)

    d = sub.add_parser("betboom-dump")
    d.add_argument("--wait", type=float, default=20)
    d.set_defaults(func=cmd_betboom_dump)

    dm = sub.add_parser("demo")
    dm.add_argument("--seed", type=int, default=7)
    dm.add_argument("--top", type=int, default=10)
    dm.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a, s)


if __name__ == "__main__":
    main()
