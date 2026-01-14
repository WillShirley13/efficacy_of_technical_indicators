CREATE TABLE ohlcv_daily (
  equity_id BIGINT UNSIGNED NOT NULL,
  trade_date DATE NOT NULL,
  open DECIMAL(18, 8),
  high DECIMAL(18, 8),
  low DECIMAL(18, 8),
  close DECIMAL(18, 8),
  adj_close DECIMAL(18, 8),
  volume BIGINT UNSIGNED,
  PRIMARY KEY (equity_id, trade_date),
  CONSTRAINT fk_ohlcv_daily_equity
    FOREIGN KEY (equity_id) REFERENCES equities (equity_id)
);
    