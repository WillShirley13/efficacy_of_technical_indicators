CREATE TABLE volume_activity_daily (
  equity_id BIGINT UNSIGNED NOT NULL,
  trade_date DATE NOT NULL,
  ti_def_id BIGINT UNSIGNED NOT NULL,
  volume_sma DECIMAL(18, 8),
  volume_ratio DECIMAL(18, 8),
  PRIMARY KEY (equity_id, trade_date, ti_def_id),
  CONSTRAINT fk_volume_activity_daily_equity
    FOREIGN KEY (equity_id) REFERENCES equities (equity_id),
  CONSTRAINT fk_volume_activity_daily_ti_def
    FOREIGN KEY (ti_def_id) REFERENCES ti_definitions (ti_def_id)
);