#!/bin/sh
# 수집기 상시 운영 설치 (sudo 없음, research R5)
#  1순위: systemd 사용자 서비스 + linger(로그인 없이 부팅 시 시작)
#  2순위: linger를 켤 수 없으면 crontab @reboot 감시 스크립트
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNIT=team2-collector.service
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$ROOT/var" "$UNIT_DIR"

"$ROOT/.venv/bin/python" -m collector check-config
"$ROOT/.venv/bin/python" -m collector init-db

sed "s#@ROOT@#$ROOT#g" "$ROOT/deploy/$UNIT" >"$UNIT_DIR/$UNIT"
systemctl --user daemon-reload

loginctl enable-linger "$USER" 2>/dev/null || true
LINGER=$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo no)

if [ "$LINGER" = "yes" ]; then
  systemctl --user enable --now "$UNIT"
  # cron 대체가 남아 있으면 지운다
  crontab -l 2>/dev/null | grep -v 'deploy/run_forever.sh' | crontab - || true
  echo "설치 완료: systemd 사용자 서비스 + linger=yes (부팅 시 로그인 없이 시작)"
  systemctl --user --no-pager status "$UNIT" | head -5
else
  # 로그인 세션에서만 도는 사용자 서비스는 cron 감시와 겹치므로 끈다
  systemctl --user disable --now "$UNIT" 2>/dev/null || true
  LINE="@reboot $ROOT/deploy/run_forever.sh"
  ( crontab -l 2>/dev/null | grep -v 'deploy/run_forever.sh'; echo "$LINE" ) | crontab -
  nohup "$ROOT/deploy/run_forever.sh" >/dev/null 2>&1 &
  echo "linger를 켤 수 없음(관리자 권한 필요) → crontab @reboot 감시 스크립트로 설치"
  echo "  등록: $LINE"
  echo "  로그: $ROOT/var/collector.log"
fi
