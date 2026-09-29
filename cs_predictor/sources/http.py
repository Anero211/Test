"""HTTP-клиент с имитацией браузера (curl_cffi, если установлен), ретраями и паузами между запросами."""
from __future__ import annotations

import logging
import random
import time
from urllib.parse import urlparse

log = logging.getLogger(__name__)

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


# Профили браузеров для curl_cffi. Cloudflare на HLTV пропускает их по-разному,
# поэтому при 403 переключаемся на следующий.
PROFILES = ["safari17_0", "edge101", "chrome", "firefox133", "safari18_0", "chrome120", "safari15_5", "edge99"]


class Fetcher:
    def __init__(self, min_interval: float = 2.5, retries: int = 12, timeout: float = 30.0):
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._last: dict[str, float] = {}
        self._profile = 0
        try:  # curl_cffi подделывает TLS-отпечаток браузера — нужно для Cloudflare на HLTV
            from curl_cffi import requests as cffi_requests
            self._cffi = cffi_requests
            self._kind = "curl_cffi"
            self._new_session()
        except ImportError:
            import httpx
            self._session = httpx.Client(headers=BROWSER_HEADERS, follow_redirects=True, timeout=timeout)
            self._kind = "httpx"

    def _new_session(self) -> None:
        self._session = self._cffi.Session(impersonate=PROFILES[self._profile % len(PROFILES)])

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        wait = self._last.get(host, 0) + self.min_interval + random.uniform(0, 0.8) - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def get(self, url: str, params: dict | None = None, headers: dict | None = None):
        h = dict(BROWSER_HEADERS, **(headers or {}))
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._throttle(url)
            try:
                if self._kind == "curl_cffi":  # заголовки ставит сам профиль браузера
                    r = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
                else:
                    r = self._session.get(url, params=params, headers=h)
            except Exception as e:  # сетевые ошибки
                last_err = e
            else:
                if r.status_code == 200:
                    return r
                last_err = RuntimeError(f"HTTP {r.status_code} для {url}")
                if r.status_code in (400, 401, 404):
                    break
                if r.status_code == 403 and self._kind == "curl_cffi":
                    self._profile += 1
                    self._new_session()
                    log.info("403 — меняю профиль браузера на %s", PROFILES[self._profile % len(PROFILES)])
                    time.sleep(3 + 2 * attempt)  # Cloudflare остывает не сразу
                    continue
            delay = 2 ** (attempt + 1)
            log.warning("%s — повтор через %ss", last_err, delay)
            time.sleep(delay)
        raise last_err or RuntimeError(url)

    def get_text(self, url: str, **kw) -> str:
        return self.get(url, **kw).text

    def get_json(self, url: str, **kw):
        return self.get(url, **kw).json()
