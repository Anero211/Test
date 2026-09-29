"""Нормализация названий команд и нечёткое сопоставление (BetBoom <-> HLTV/PandaScore)."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

# Слова, которые разные источники то пишут, то нет: "Team Spirit" == "Spirit"
_STOPWORDS = {"team", "esports", "esport", "e-sports", "gaming", "club", "gg", "cs", "cs2", "csgo", "org"}

# Встроенные синонимы (ключ и значение — уже нормализованные). Можно расширить data/aliases.json
BUILTIN_ALIASES = {
    "navi": "natusvincere", "нави": "natusvincere", "natusvincere": "natusvincere",
    "vp": "virtuspro", "virtuspro": "virtuspro", "виртуспро": "virtuspro",
    "mousesports": "mouz", "mouz": "mouz",
    "col": "complexity", "complexity": "complexity",
    "ef": "eternalfire", "eternalfire": "eternalfire",
    "mongolz": "themongolz", "themongolz": "themongolz",
    "big": "big", "berlininternationalgaming": "big",
    "gl": "gamerlegion", "gamerlegion": "gamerlegion",
    "3dmax": "3dmax", "spirit": "spirit", "спирит": "spirit",
    "бетбум": "betboom", "betboom": "betboom",
    "faze": "faze", "fazeclan": "faze",
    "g2": "g2", "vitality": "vitality", "vit": "vitality",
    "liquid": "liquid", "tl": "liquid",
    "heroic": "heroic", "astralis": "astralis", "furia": "furia", "falcons": "falcons",
    "pain": "pain", "aurora": "aurora", "mibr": "mibr", "nemiga": "nemiga", "b8": "b8",
}


def normalize(name: str) -> str:
    """'Team Spirit' -> 'spirit', 'Natus Vincere' -> 'natusvincere', 'MOUZ' -> 'mouz'."""
    s = unicodedata.normalize("NFKC", name or "").lower().strip()
    s = s.replace("&", " and ").replace(".", "")
    words = [w for w in re.split(r"[^0-9a-zа-яё]+", s) if w and w not in _STOPWORDS]
    if not words:  # напр. название целиком "Team"
        words = [w for w in re.split(r"[^0-9a-zа-яё]+", s) if w]
    return "".join(words)


class TeamMatcher:
    """Сопоставляет произвольное название с каноническими именами из истории матчей."""

    def __init__(self, canonical_names: list[str], aliases_path: Path | None = None, cutoff: float = 86.0):
        self.cutoff = cutoff
        self.aliases = dict(BUILTIN_ALIASES)
        if aliases_path and aliases_path.exists():
            extra = json.loads(aliases_path.read_text(encoding="utf-8"))
            self.aliases.update({normalize(k): normalize(v) for k, v in extra.items()})
        self.by_key: dict[str, str] = {}
        for name in canonical_names:
            self.by_key.setdefault(self._key(name), name)
        self._keys = list(self.by_key)

    def _key(self, name: str) -> str:
        k = normalize(name)
        return self.aliases.get(k, k)

    def match(self, name: str) -> tuple[str | None, float]:
        """Возвращает (каноническое имя, уверенность 0..100) или (None, score)."""
        k = self._key(name)
        if k in self.by_key:
            return self.by_key[k], 100.0
        if not self._keys:
            return None, 0.0
        best = process.extractOne(k, self._keys, scorer=fuzz.WRatio)
        if best is None:
            return None, 0.0
        key, score, _ = best
        # Защита от "spirit" -> "spiritacademy": длина ключей должна быть близкой
        if score >= self.cutoff and fuzz.ratio(k, key) >= 70:
            return self.by_key[key], float(score)
        return None, float(score)
