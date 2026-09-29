"""Автоматический бот: линия BetBoom → модель → сигнал «СТАВИТЬ» в Telegram → журнал → ROI.

Цикл (раз в `interval` секунд):
  1. раз в час: докачать свежие результаты HLTV, переобучить модель, рассчитать открытые ставки;
  2. забрать линию CS2 с BetBoom и сравнить каждый матч с моделью;
  3. по новым сигналам — сообщение в Telegram и запись в журнал (data/bets.csv);
  4. раз в день — сводка журнала (ROI) в Telegram.
"""
from __future__ import annotations

import html
import logging
import random
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import journal
from .config import DATA_DIR, Settings
from .names import TeamMatcher
from .notify import send_telegram
from .signals import Row, evaluate, log_odds
from .sources import betboom
from .storage import load_matches

log = logging.getLogger(__name__)
BETS = DATA_DIR / "bets.csv"
MSK = ZoneInfo("Europe/Moscow")


def signal_text(r: Row, stake: float) -> str:
    e = html.escape
    side = "П1" if r.pick == r.team1 else "П2"
    book_p = r.book_p1 if side == "П1" else 1 - r.book_p1
    start = datetime.fromisoformat(r.start).astimezone(MSK).strftime("%d.%m %H:%M") if r.start else "—"
    return (
        f"✅ <b>СТАВИТЬ: {side} {e(r.pick)} @ {r.pick_odds:.2f}</b>\n"
        f"{e(r.team1)} vs {e(r.team2)} · Bo{r.best_of} · {start} МСК\n"
        f"Модель: {r.pick_p:.0%} · BetBoom считает: {book_p:.0%}\n"
        f"Перевес (EV): <b>{r.pick_ev:+.1%}</b> · ставка: <b>{stake:g}% банка</b>\n"
        f"<i>Проверь составы на HLTV перед ставкой</i>"
    )


def settled_text(done: list[journal.Bet], bets: list[journal.Bet]) -> str:
    lines = ["📋 <b>Рассчитаны ставки</b>"]
    for b in done:
        icon = {"won": "✅", "lost": "❌"}.get(b.status, "➖")
        lines.append(f"{icon} {html.escape(b.team1)} – {html.escape(b.team2)} {b.score}: "
                     f"{b.market} {html.escape(b.pick)} @ {b.odds:.2f} → {b.profit:+.2f}%")
    st = journal.stats(bets)
    if st["settled"]:
        lines.append(f"Всего: {st['won']}/{st['settled']}, прибыль {st['profit']:+.2f}% банка, ROI {st['roi']:+.1%}")
    return "\n".join(lines)


class Bot:
    def __init__(self, s: Settings, mode: str, interval: int, stake: float, dry_run: bool,
                 summary_hour: int = 10, refresh: bool = True):
        self.s, self.mode, self.interval, self.stake, self.dry = s, mode, interval, stake, dry_run
        self.summary_hour = summary_hour
        self.refresh = refresh
        self.predictor = None
        self.matcher = None
        self.last_train = 0.0
        self.schedule: list[dict] = []
        self.last_schedule = 0.0
        self.last_summary_day = None

    # ---------- вспомогательное ----------
    def notify(self, text: str) -> None:
        if self.dry or not (self.s.telegram_token and self.s.telegram_chat_id):
            print("\n[Telegram]\n" + text.replace("<b>", "").replace("</b>", "")
                  .replace("<i>", "").replace("</i>", ""))
            return
        send_telegram(self.s.telegram_token, self.s.telegram_chat_id, text)

    def retrain(self, refresh: bool) -> None:
        from .__main__ import _load_predictor
        self.predictor = _load_predictor(self.s, refresh)
        self.matcher = TeamMatcher(sorted(self.predictor.engine.rating), self.s.aliases_json)
        self.last_train = time.time()
        bets = journal.load(BETS)
        done = journal.settle(bets, load_matches(self.s.matches_csv))
        if done:
            journal.save(BETS, bets)
            self.notify(settled_text(done, bets))

    def get_schedule(self) -> list[dict]:
        if time.time() - self.last_schedule > 1800:  # расписание HLTV — не чаще раза в 30 минут
            from .__main__ import _schedule
            self.schedule = _schedule(self.s, self.s.horizon_hours)
            self.last_schedule = time.time()
        return self.schedule

    # ---------- один проход ----------
    def tick(self) -> list[Row]:
        if self.predictor is None or time.time() - self.last_train > 3600:
            self.retrain(self.refresh)
        line = betboom.fetch_line(self.mode, self.s.betboom_page_url, self.s.betboom_json_url,
                                  self.s.betboom_manual_csv)
        if not line:
            log.warning("линия BetBoom пуста")
            return []
        log_odds(line, self.s.odds_log_csv)
        need = any(ev.best_of is None for ev in line)
        rows = evaluate(line, self.predictor, self.matcher, self.get_schedule() if need else [], self.s)
        now = datetime.now(timezone.utc)
        bets = journal.load(BETS)
        new = 0
        for r in rows:
            if not r.signal or not r.start or datetime.fromisoformat(r.start) <= now:
                continue  # только будущие матчи с известным временем начала
            pick = r.model_team1 if r.pick == r.team1 else r.model_team2
            bet = journal.Bet(
                bet_id=f"{now:%Y%m%d%H%M%S}-{new}", created_at=now.isoformat(), start=r.start,
                team1=r.model_team1, team2=r.model_team2, best_of=r.best_of or 0, pick=pick,
                market="П1" if r.pick == r.team1 else "П2", odds=r.pick_odds, p_model=round(r.pick_p, 4),
                ev=round(r.pick_ev, 4), stake=self.stake)
            if journal.add_if_new(bets, bet):
                new += 1
                self.notify(signal_text(r, self.stake))
        if new:
            journal.save(BETS, bets)
        log.info("линия: %s матчей, сигналов: %s, новых: %s", len(rows), sum(r.signal for r in rows), new)
        self.maybe_summary(bets)
        return rows

    def maybe_summary(self, bets: list[journal.Bet]) -> None:
        local = datetime.now(MSK)
        if local.hour >= self.summary_hour and self.last_summary_day != local.date():
            self.last_summary_day = local.date()
            self.notify("📊 <b>Сводка за всё время</b>\n" + html.escape(journal.report(bets)))

    def run(self, once: bool = False) -> None:
        self.last_summary_day = datetime.now(MSK).date() if once else None
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("ошибка в цикле бота")
            if once:
                return
            time.sleep(self.interval + random.uniform(0, 20))
