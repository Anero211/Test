"""Отправка сигналов в Telegram через Bot API (без тяжёлых зависимостей)."""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import httpx

from .signals import Row

log = logging.getLogger(__name__)


def format_signal(r: Row) -> str:
    e = html.escape
    return (
        f"🎯 <b>{e(r.team1)} vs {e(r.team2)}</b> (Bo{r.best_of})\n"
        f"🏆 {e(r.tournament or '—')}\n"
        f"🕒 {r.start[:16].replace('T', ' ')} UTC\n"
        f"Ставка: <b>П{1 if r.pick == r.team1 else 2} — {e(r.pick)}</b> @ {r.pick_odds:.2f} (BetBoom)\n"
        f"Модель: {r.pick_p:.1%} | БК без маржи: "
        f"{(r.book_p1 if r.pick == r.team1 else 1 - r.book_p1):.1%}\n"
        f"EV: <b>{r.pick_ev:+.1%}</b> | 1/4 Келли: {r.kelly:.1%} банка"
    )


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    try:
        r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                       json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                             "disable_web_page_preview": True})
        if r.status_code != 200:
            log.warning("Telegram: %s %s", r.status_code, r.text[:200])
        return r.status_code == 200
    except httpx.HTTPError as e:
        log.warning("Telegram: %s", e)
        return False


def send_new_signals(rows: list[Row], token: str, chat_id: str, sent_path: Path) -> int:
    """Шлёт только новые сигналы (повторно — если кэф изменился заметно)."""
    sent = json.loads(sent_path.read_text()) if sent_path.exists() else {}
    n = 0
    for r in rows:
        if not r.signal:
            continue
        key = f"{r.start[:10]}|{r.team1}|{r.team2}|{r.pick}"
        if key in sent and abs(sent[key] - r.pick_odds) < 0.05:
            continue
        if send_telegram(token, chat_id, format_signal(r)):
            sent[key] = r.pick_odds
            n += 1
    sent_path.parent.mkdir(parents=True, exist_ok=True)
    sent_path.write_text(json.dumps(sent, ensure_ascii=False, indent=1))
    return n
