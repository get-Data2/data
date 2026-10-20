-- 과제 1 수집 저장소 DDL (MySQL 8.0, cf_u36_d3). 시각은 모두 UTC.
-- `init-db`가 {p} 자리에 storage.table_prefix를 넣어 실행한다(멱등).

CREATE TABLE IF NOT EXISTS {p}space (
  space_id   VARCHAR(32)  NOT NULL PRIMARY KEY,
  name       VARCHAR(64)  NOT NULL,
  kind       VARCHAR(16)  NOT NULL,
  area_m2    DOUBLE       NULL,
  note       VARCHAR(256) NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS {p}device (
  device_id         VARCHAR(64)  NOT NULL PRIMARY KEY,
  source            VARCHAR(32)  NOT NULL,
  name              VARCHAR(128) NULL,
  model             VARCHAR(64)  NULL,
  space_id          VARCHAR(32)  NOT NULL,
  spot              VARCHAR(128) NULL,
  raw_location      VARCHAR(128) NULL,
  mapped_by         VARCHAR(16)  NOT NULL,
  expected_period_s INT          NOT NULL DEFAULT 60,
  first_seen        DATETIME(3)  NOT NULL,
  last_seen         DATETIME(3)  NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS {p}measurement (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  ts           DATETIME(3)  NOT NULL,
  space_id     VARCHAR(32)  NOT NULL,
  device_id    VARCHAR(64)  NOT NULL,
  item         VARCHAR(64)  NOT NULL,
  value        DOUBLE       NOT NULL,
  unit         VARCHAR(16)  NULL,
  source       VARCHAR(32)  NOT NULL,
  received_at  DATETIME(3)  NOT NULL,
  source_ts    DATETIME(3)  NULL,
  ts_corrected TINYINT(1)   NOT NULL DEFAULT 0,
  frame_cnt    INT UNSIGNED NULL,
  UNIQUE KEY uq_meas (device_id, item, ts),
  KEY ix_space_item_ts (space_id, item, ts),
  KEY ix_ts (ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS {p}collector_event (
  id     BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  ts     DATETIME(3)  NOT NULL,
  kind   VARCHAR(32)  NOT NULL,
  detail VARCHAR(512) NULL,
  host   VARCHAR(64)  NOT NULL,
  KEY ix_ts (ts)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS {p}ingest_stats (
  minute    DATETIME    NOT NULL,
  host      VARCHAR(64) NOT NULL,
  messages  INT         NOT NULL,
  `stored`  INT         NOT NULL,
  ignored   INT         NOT NULL,
  failed    INT         NOT NULL,
  duplicate INT         NOT NULL,
  max_lag_s DOUBLE      NULL,
  PRIMARY KEY (minute, host)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
