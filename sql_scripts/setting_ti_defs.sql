INSERT INTO ti_definitions (ti_name, time_frame, params_json)
VALUES
    -- RSI
    ('RSI', 3, '{"length": 7}'),
    ('RSI', 10, '{"length": 14}'),
    ('RSI', 20, '{"length": 14}'),

    -- Stochastic
    ('STOCH', 3, '{"k": 7, "d": 3}'), 
    ('STOCH', 10, '{"k": 14, "d": 3}'),
    ('STOCH', 20, '{"k": 14, "d": 3}'),

    -- SMA
    ('SMA', 3, '{"sma_fast": 10, "sma_slow": 20}'),
    ('SMA', 10, '{"sma_fast": 20, "sma_slow": 50}'),
    ('SMA', 20, '{"sma_fast": 50, "sma_slow": 100}'),

    -- EMA
    ('EMA', 3, '{"ema_fast": 5, "ema_slow": 13}'),
    ('EMA', 10, '{"ema_fast": 12, "ema_slow": 26}'),
    ('EMA', 20, '{"ema_fast": 12, "ema_slow": 26}'),

    -- MACD
    ('MACD', 3, '{"fast": 5, "slow": 13, "signal": 5}'),
    ('MACD', 10, '{"fast": 12, "slow": 26, "signal": 9}'),
    ('MACD', 20, '{"fast": 12, "slow": 26, "signal": 9}'),

    -- ADX
    ('ADX', 3, '{"length": 7}'),
    ('ADX', 10, '{"length": 14}'),
    ('ADX', 20, '{"length": 14}'),

    -- ATR
    ('ATR', 3, '{"length": 7}'),
    ('ATR', 10, '{"length": 14}'),
    ('ATR', 20, '{"length": 14}'),

    -- Bollinger Bands
    ('BBANDS', 3, '{"length": 10, "std": 2}'),
    ('BBANDS', 10, '{"length": 20, "std": 2}'),
    ('BBANDS', 20, '{"length": 20, "std": 2}'),

    -- OBV
    ('OBV', 3, '{"signal": 10}'),
    ('OBV', 10, '{"signal": 20}'),
    ('OBV', 20, '{"signal": 20}'),

    -- Volume Rate of Change
    ('VOL_ROC', 3, '{"length": 10}'),
    ('VOL_ROC', 10, '{"length": 20}'),
    ('VOL_ROC', 20, '{"length": 20}');