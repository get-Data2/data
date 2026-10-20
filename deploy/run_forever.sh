#!/bin/sh
# linger를 켤 수 없을 때 쓰는 감시 스크립트: crontab @reboot 로 실행된다.
# 이미 수집기가 돌고 있으면(잠금 파일) 바로 끝나고, 아니면 수집기가 죽을 때마다 5초 뒤 다시 띄운다.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/var"
LOG="$ROOT/var/collector.log"
exec 9>"$ROOT/var/collector.lock"
flock -n 9 || exit 0
cd "$ROOT" || exit 1
while :; do
  if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 10485760 ]; then
    mv -f "$LOG" "$LOG.1"
  fi
  echo "$(date -Is) 수집기 시작" >>"$LOG"
  # 잠금(9)은 자식에게 넘기지 않는다 — 감시 스크립트가 잠금을 쥔다
  "$ROOT/.venv/bin/python" -m collector run 9>&- >>"$LOG" 2>&1
  echo "$(date -Is) 수집기 종료 코드 $? — 5초 뒤 재시작" >>"$LOG"
  sleep 5
done
