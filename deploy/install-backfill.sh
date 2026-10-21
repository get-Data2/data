#!/bin/sh
# InfluxDB 과거 데이터 적재 타이머 설치(sudo 없음). 다시 처음부터 확인하려면 var/backfill_done을 지우고 다시 실행.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$ROOT/var" "$UNIT_DIR"
for u in wonil-backfill.service wonil-backfill.timer; do
  sed "s#@ROOT@#$ROOT#g" "$ROOT/deploy/$u" >"$UNIT_DIR/$u"
done
systemctl --user daemon-reload
systemctl --user enable --now wonil-backfill.timer
systemctl --user list-timers wonil-backfill.timer --no-pager | head -3
