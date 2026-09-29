"""Настройки: читаются из переменных окружения и файла .env в корне проекта."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Минимальный загрузчик .env без внешних зависимостей."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


@dataclass
class Settings:
    # Источники
    pandascore_token: str = ""
    betboom_page_url: str = "https://betboom.ru/sport/esports/counter-strike"
    betboom_json_url: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Фильтр сигналов (из ТЗ)
    min_odds: float = 1.55
    max_odds: float = 1.85
    min_ev: float = 0.07
    max_ev: float = 0.30  # выше — скорее ошибка модели (стендины, свежая замена), чем реальный перевес
    allowed_formats: tuple[int, ...] = (3,)  # только Bo3
    horizon_hours: float = 48.0
    min_team_games: int = 15  # не доверяем командам с короткой историей

    # Файлы
    matches_csv: Path = field(default_factory=lambda: DATA_DIR / "matches.csv")
    odds_log_csv: Path = field(default_factory=lambda: DATA_DIR / "odds_log.csv")
    betboom_manual_csv: Path = field(default_factory=lambda: DATA_DIR / "betboom_manual.csv")
    aliases_json: Path = field(default_factory=lambda: DATA_DIR / "aliases.json")
    sent_json: Path = field(default_factory=lambda: DATA_DIR / "sent_signals.json")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        s = cls()
        s.pandascore_token = _env("PANDASCORE_TOKEN")
        s.betboom_page_url = _env("BETBOOM_CS2_URL", s.betboom_page_url)
        s.betboom_json_url = _env("BETBOOM_JSON_URL")
        s.telegram_token = _env("TELEGRAM_BOT_TOKEN")
        s.telegram_chat_id = _env("TELEGRAM_CHAT_ID")
        s.min_odds = float(_env("MIN_ODDS", str(s.min_odds)))
        s.max_odds = float(_env("MAX_ODDS", str(s.max_odds)))
        s.min_ev = float(_env("MIN_EV", str(s.min_ev)))
        s.max_ev = float(_env("MAX_EV", str(s.max_ev)))
        s.horizon_hours = float(_env("HORIZON_HOURS", str(s.horizon_hours)))
        return s
