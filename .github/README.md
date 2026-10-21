# 시계열 데이터 추론 실습 — team2 (코드)

교육장 센서(CO₂·온도·습도)와 냉난방 데이터로 **"지금 무슨 일이 일어나고 있고, 곧 어떻게 될까"**에 답하는 추론 기능을
만들어 팀 Jetson에서 상시 운영한다.

> 📄 **계획과 결과 문서는 [get-Data2/data-docs](https://github.com/get-Data2/data-docs)**에 있다.
> 과제 안내서 원문은 [README.md](../README.md)(읽기 전용)다.

## 과제 진행

| 과제 | 상태 | 계획(spec) | 코드 |
|---|---|---|---|
| 1. 데이터 수집 체계 | 🟢 완료·상시 수집 중 (2026-10-20~), 24h 최종표만 남음 — [산출물](https://github.com/get-Data2/data-docs/tree/main/docs/deliverables/task1) · [점검](https://github.com/get-Data2/data-docs/blob/main/docs/tasks/task1-data-collection.md) | [001-data-collection](https://github.com/get-Data2/data-docs/tree/main/specs/001-data-collection) | `src/collector/` |
| 2. 데이터 이해와 기준선 | 🟢 완료 — 분할 `v1`이 과제 3~6 평가 기준 — [산출물](https://github.com/get-Data2/data-docs/tree/main/docs/deliverables/task2) · [점검](https://github.com/get-Data2/data-docs/blob/main/docs/tasks/task2-data-baseline.md) | [002-data-baseline](https://github.com/get-Data2/data-docs/tree/main/specs/002-data-baseline) | `src/analysis/` |
| 3. 60분 예측 | ⚪ 대기 | — | — |
| 4. 이상 탐지 | ⚪ 대기 | — | — |
| 5. 상태 추정 | ⚪ 대기 | — | — |
| 6. 사전학습 모델 비교 | ⚪ 대기 | — | — |
| 7. 엣지 최적화 | ⚪ 대기 | — | — |
| 8. 상시 추론 서비스 | ⚪ 대기 | — | — |

## 빠른 사용 (과제 1 수집기)

아래 명령은 프로젝트 폴더(`~/Desktop/project`)에서 입력한다. 어디서든 쓰려면 `.venv/bin/python -m collector` 대신 `collector`(→ `tools/collector`).
서비스 이름은 `wonil-collector`(systemd 사용자 서비스).

```bash
~/.local/bin/uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m collector check-config         # 설정 검증
.venv/bin/python -m collector check                # 연결 점검(서비스·MQTT·MySQL·InfluxDB)
deploy/install.sh                                  # 상시 서비스 설치(systemd 사용자 서비스 + linger)
.venv/bin/python -m collector report --hours 24    # 수집 현황(커버율·빠진 구간·원인 단서)
.venv/bin/python -m pytest                         # 단위 + 시험 브로커 (-m mysql / -m live 별도)
```

## 빠른 사용 (과제 2 분석)

```bash
~/.local/bin/uv pip install --python .venv/bin/python -e '.[test,analysis]'
analysis all          # MySQL → 사본 → 정제 → 분할 → 사건 → 기준선 → 진단서 (약 4분, 산출물 docs/deliverables/task2/)
analysis info         # 파일·해시 상태(기준선과 평가 시점 일치 확인)
```

## 구조

```text
src/<패키지>/     소스 코드 (과제별 패키지, 예: src/collector/)
config/           동작 설정 YAML — 형식 변경·센서 추가·공간 매핑은 여기만 고친다
deploy/           systemd 사용자 서비스·설치 스크립트
tests/            pytest (unit / broker / mysql / live 마커)
tools/            보조 도구 (mqtt_ping.py, docs-git)
.specify/         Spec Kit 설정·템플릿·헌법
```

## 환경

- Jetson (L4T R39, aarch64, RAM 7.3GB), Python 3.12. sudo·Docker 없음
- 패키지: `~/.local/bin/uv pip install --python .venv/bin/python <pkg>`. 실행은 `.venv/bin/python`
- 접속 정보는 `.env`(커밋하지 않음). 키 목록: `MQTT_*`, `MYSQL_*`
- 개발 흐름: Spec Kit `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`

## 저장소 두 개, 작업 폴더 하나

| 저장소 | 추적 대상 | 명령 |
|---|---|---|
| `get-Data2/data` (이곳) | 코드·설정·배포·시험 | `git …` |
| `get-Data2/data-docs` | `specs/`, `docs/`, 헌법 | `tools/docs-git …` |
