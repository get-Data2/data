#!/bin/sh
# 상시 추론 서비스(과제 8) 설치 — sudo 없음. 보드 로컬 브로커(wonil-broker)와 서비스(wonil-inference)를 사용자 서비스로 등록한다.
# Linger=yes는 과제 1에서 적용했다(재부팅 시 로그인 없이 시작).
# 실행 전: .env에 INFER_MQTT_USER, INFER_MQTT_PASS를 추가한다(값은 화면에 출력하지 않는다).
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$ROOT/var/service" "$UNIT_DIR"

"$ROOT/.venv/bin/python" -c "from service.config import load_service_config; load_service_config()" \
  || { echo "config/service.yaml 오류"; exit 2; }
for k in INFER_MQTT_USER INFER_MQTT_PASS; do
  cut -d= -f1 "$ROOT/.env" | grep -qx "$k" || { echo ".env에 $k 키가 없음 — 추가한 뒤 다시 실행"; exit 2; }
done
for p in 11883 18080; do
  if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -q ":$p\$"; then
    echo "포트 $p 사용 중 — config/service.yaml(broker.port·api.port)을 바꾸거나 점유 프로세스 확인"; exit 3
  fi
done

for UNIT in wonil-broker.service wonil-inference.service; do
  sed "s#@ROOT@#$ROOT#g" "$ROOT/deploy/$UNIT" >"$UNIT_DIR/$UNIT"
done
systemctl --user daemon-reload
systemctl --user enable --now wonil-broker.service
sleep 2
systemctl --user enable --now wonil-inference.service
echo "설치 완료: wonil-broker(11883), wonil-inference(조회 API 18080)"
systemctl --user --no-pager status wonil-broker.service wonil-inference.service | grep -E "●|Active:"
echo "확인: infer status  /  curl -s http://127.0.0.1:18080/v1/health"
