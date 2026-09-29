#!/usr/bin/env bash
# Установка бота на сервер Ubuntu 22.04/24.04. Запуск из папки проекта:  sudo bash deploy/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$(pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"

echo "==> Системные пакеты"
apt-get update -y
apt-get install -y python3-venv python3-pip git

# Браузеру для линии BetBoom нужна память: если RAM < 2 ГБ и нет swap — добавляем 2 ГБ swap
if [ "$(free -m | awk '/Mem:/{print $2}')" -lt 1900 ] && [ "$(swapon --show | wc -l)" -eq 0 ]; then
  echo "==> Добавляю swap 2 ГБ"
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> Python-окружение"
sudo -u "$RUN_USER" python3 -m venv .venv
sudo -u "$RUN_USER" .venv/bin/pip install -q --upgrade pip
sudo -u "$RUN_USER" .venv/bin/pip install -q -r requirements.txt

echo "==> Chromium для чтения линии BetBoom"
.venv/bin/playwright install-deps chromium
sudo -u "$RUN_USER" .venv/bin/playwright install chromium

[ -f .env ] || { sudo -u "$RUN_USER" cp .env.example .env; echo "==> Создан .env — впиши TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID"; }

echo "==> Служба systemd (автозапуск и перезапуск при сбоях)"
sed -e "s#__DIR__#$DIR#g" -e "s#__USER__#$RUN_USER#g" deploy/cs-predictor-bot.service \
  > /etc/systemd/system/cs-predictor-bot.service
systemctl daemon-reload

cat <<MSG

Готово. Дальше:
  1) заполни .env (Telegram) и проверь один проход без отправки:
       .venv/bin/python -m cs_predictor bot --once --dry-run
  2) включи бота:
       sudo systemctl enable --now cs-predictor-bot
  3) логи:            journalctl -u cs-predictor-bot -f
     журнал ставок:   .venv/bin/python -m cs_predictor journal
MSG
