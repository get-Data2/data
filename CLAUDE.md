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
- 상시 실행: systemd 사용자 서비스. `Linger=no`이므로 재부팅 자동 시작은 `loginctl enable-linger` 또는 `crontab @reboot`로 보완.

## 외부 자원 (접속 정보는 `.env`)
| 자원 | 용도 | 권한 |
|---|---|---|
| MQTT 브로커 | 실시간 센서 수신 (ChirpStack LoRaWAN, xflow 냉난방) | 구독 |
| MySQL 8.0 (`cf_u36_d3`) | **수집 데이터 저장소** | 팀 DB 내 DDL/DML |
| InfluxDB (사설 IP) | 원본 대조용 | 조회 전용, 현재 네트워크에서 접근 불가 |

## 개발 방식
- Spec Kit: `/speckit-constitution` → `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`
- 과제 하나 = spec 하나(`specs/NNN-이름/`). README의 "해결할 것" → 요구사항, "확인 기준" → 수용 기준.
- 테스트: `pytest`. 확인 기준(재접속 1분 내 등)은 자동 테스트로 검증한다.
- 결과표에는 측정 조건(데이터 기간, 실/합성, 시험 기간, 전력 모드)을 함께 적는다.

## 디렉토리
| 위치 | 역할 |
|---|---|
| `specs/NNN-이름/` | 계획(Spec Kit: spec·plan·tasks) |
| `src/<패키지>/` | 소스 코드 (과제별 패키지, 예: `src/collector/`) |
| `config/` | 동작 설정(YAML, 비밀 없음) |
| `deploy/` | systemd 사용자 서비스·설치 스크립트 |
| `tests/` | pytest (`unit`/`broker`/`mysql`/`live` 마커) |
| `docs/` | 결과·운영 문서, `docs/reports/` 측정 결과 |
| `var/` | 실행 산출물(스풀·로그), git 제외 |

## Git (작업 폴더 하나, 저장소 둘)
- **코드** `get-Data2/data`: 일반 `git`. `specs/`·`docs/`는 `.git/info/exclude`로 제외.
- **계획·문서** `get-Data2/data-docs`: `tools/docs-git <명령>`(= `.git-docs`). `specs/`, `docs/`, 헌법만 추적.
- 두 저장소 모두 **공개**다. 커밋 전 비밀 값·내부 주소가 없는지 확인한다.
- 저장소 첫 화면: `data`는 `.github/README.md`, `data-docs`는 `docs/README.md` (GitHub 우선순위 `.github` → 루트 → `docs`).
  보호 훅은 이름에 `README.md`가 든 셸 쓰기 명령을 막으므로 이 파일들은 Write/Edit 도구로 고친다.

## 진행 상황
- [ ] 과제 1 데이터 수집 체계
