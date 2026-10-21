# 시계열 데이터 추론 실습 (team2 Jetson)

## 절대 규칙
- **답변·문서는 모두 한국어.** 코드 식별자·명령어만 원문 유지.
- **`README.md`는 읽기 전용**(과제 안내서). 수정·덮어쓰기·이동 금지. 원본 해시: `.backup/README.sha256`.
- **`.env` 내용은 출력 금지.** 코드에서는 `python-dotenv`로 로드만 한다. 키 목록은 `cut -d= -f1 .env`.
- 위 두 규칙은 `.claude/hooks/protect.py`(PreToolUse 훅)가 강제한다.

## 환경
- Jetson (L4T R39, aarch64, RAM 7.3GB), Python 3.12.3. **sudo·Docker 권한 없음.**
- 패키지는 `~/.local/bin/uv`로 `.venv`에 설치: `uv pip install --python .venv/bin/python <pkg>`
- 실행은 항상 `.venv/bin/python`.
- 상시 실행: systemd 사용자 서비스. `Linger=yes`(2026-10-20 sudo 없이 `loginctl enable-linger` 적용) → 재부팅 시 로그인 없이 시작.

## 외부 자원 (접속 정보는 `.env`)
| 자원 | 용도 | 권한 |
|---|---|---|
| MQTT 브로커 | 실시간 센서 수신 (ChirpStack LoRaWAN, xflow 냉난방) | 구독 |
| MySQL 8.0 (`cf_u36_d3`) | **수집 데이터 저장소** | 팀 DB 내 DDL/DML |
| InfluxDB (사설 IP) | 과거 데이터(2025-08~) 적재 원천 | 조회 전용. **유선 랜 경로(10.x)가 있을 때만** 접근됨 |

## 개발 방식
- Spec Kit: `/speckit-constitution` → `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`
- 과제 하나 = spec 하나(`specs/NNN-이름/`). README의 "해결할 것" → 요구사항, "확인 기준" → 수용 기준.
- 테스트: `pytest`. 확인 기준(재접속 1분 내 등)은 자동 테스트로 검증한다.
- 결과표에는 측정 조건(데이터 기간, 실/합성, 시험 기간, 전력 모드)을 함께 적는다.

## 디렉토리
| 위치 | 역할 |
|---|---|
| `specs/NNN-이름/` | 계획(Spec Kit: spec·plan·tasks) |
| `src/<패키지>/` | 소스 코드 (과제별 패키지: `src/collector/` 과제 1, `src/analysis/` 과제 2, `src/forecast/` 과제 3) |
| `config/` | 동작 설정(YAML, 비밀 없음) |
| `deploy/` | systemd 사용자 서비스·설치 스크립트 |
| `tests/` | pytest (`unit`/`broker`/`mysql`/`live` 마커) |
| `docs/` | 문서(아래 하위 폴더). 첫 화면 `docs/README.md` |
| `docs/deliverables/taskN/` | **산출물만**(README 각 과제의 "산출물" 항목: 보고서·표·그래프). 목차 `docs/deliverables/README.md` |
| `docs/tasks/taskN-*.md` | **과제별 점검 문서**. README의 해결할 것·확인 기준·생각해 볼 점·심화(B)를 항목마다 대응시키고 결과·측정 조건·해석을 채운다. 산출물 본문은 넣지 않고 `deliverables/`로 연결. **기능별 확인 명령(명령·입력 위치·정상 결과)**을 실제로 실행해 맞춘 뒤 적는다. 과제를 마칠 때 반드시 갱신 |
| `docs/ops/`, `docs/advanced/taskN/`, `docs/reports/` | 운영 가이드 / 심화(B) 결과 / 측정·작업 기록(자동 생성 상세 표 포함) |
| `var/` | 실행 산출물(스풀·로그), git 제외 |

## Git (작업 폴더 하나, 저장소 둘)
- **코드** `get-Data2/data`: 일반 `git`. `specs/`·`docs/`는 `.git/info/exclude`로 제외.
- **계획·문서** `get-Data2/data-docs`: `tools/docs-git <명령>`(= `.git-docs`). `specs/`, `docs/`, 헌법만 추적.
- 두 저장소 모두 **공개**다. 커밋 전 비밀 값·내부 주소가 없는지 확인한다.
- 저장소 첫 화면: `data`는 `.github/README.md`, `data-docs`는 `docs/README.md` (GitHub 우선순위 `.github` → 루트 → `docs`).
  보호 훅은 이름에 `README.md`가 든 셸 쓰기 명령을 막으므로 이 파일들은 Write/Edit 도구로 고친다.

## 진행 상황
- [x] 과제 1 데이터 수집 체계 — 완료·상시 수집 중(2026-10-20 16:33~). 24h 최종 커버율 최저 98.6%(2026-10-21). 남은 것: InfluxDB 미적재 90일·장치 미특정 75만 행(보드 유선 링크 복구 또는 맥 CSV 내보내기 `collector influx-export-script` → `backfill --from-dir`). 연결 점검 `collector check`
- [x] 과제 2 데이터 이해와 기준선 — 완료(2026-10-21). 분할 `v1-provisional`이 과제 3~6 공식 평가 기준(`var/eval/eval_points_v1-provisional.parquet`).
  `v2-final`(2학기 시험)은 InfluxDB 적재 후 추가 비교(선택). 산출물 `docs/deliverables/task2/`, 점검 `docs/tasks/task2-data-baseline.md`
- [ ] 과제 3 60분 예측 — spec·plan 완료(2026-10-21). GPU(Orin, torch cu130) 사용, 그룹 미반영 세션은 `sg video`/`sg render`. 계획 `specs/003-forecast-60min/`
- [ ] 과제 8 상시 추론 서비스 — 상시 가동 중(`wonil-inference` + 보드 브로커 `wonil-broker` :11883, API :18080, CLI `infer`).
  장애 시험·API·결과 형식 문서·심화 2개 완료(2026-10-21). 남은 것: 12시간 표(10-22 08:25 자동 생성 → `docs/deliverables/task8/tables/02-12h-run.md`로 산출물 ② 마무리), 재부팅 실측(사용자 일정).
  서비스 DB 읽기는 **자동 커밋 필수**(끄면 REPEATABLE READ 스냅샷에 갇혀 새 행을 못 봄). 모델 교체는 `infer reload`(PID 파일로 파이썬에만 HUP)
