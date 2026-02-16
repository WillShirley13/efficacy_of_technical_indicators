import json
from typing import cast

import pandas as pd
import pandas_ta  # noqa: F401
from mysql.connector.abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
from mysql.connector.pooling import PooledMySQLConnection

from src.utils import get_db_conn, get_equity_id, get_stock_data


def _get_pandas_ta_col_name(df: pd.DataFrame, prefix: str) -> str:
    """
    Pick exactly one column from `df` whose name starts with `prefix`.

    """
    matches: list[str] = [str(c) for c in df.columns if str(c).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 column starting with {prefix!r}, got {matches}")
    return matches[0]


def _get_ti_definition(
    cursor: MySQLCursorAbstract,
    ti_name: str,
    timeframe: int,
) -> tuple[int, dict[str, int]]:
    """
    Get technical indicator + timeframe-specific database id and corresponding params
    """
    cursor.execute(
        "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
        (ti_name, timeframe),
    )
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
    ti_def_id: int = int(result[0])  # type: ignore[arg-type]
    params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
    return ti_def_id, params_json


def derive_rsi(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive RSI values and upsert them into `rsi_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-days.
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "RSI"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    length: int = params_json["length"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving relative strength index values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: length={length}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float]] = []

        print(f"  → Calculating RSI with length={length}...")
        rsi: pd.Series = df.ta.rsi(close="close", length=length)

        #  Prepare rows
        for date, rsi_value in rsi.iloc[length:].items():
            # Skip warm-up NaNs and any missing values (avoids MySQL insert issues).
            if pd.isna(rsi_value):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            rsi_daily_row = (equity_id, date_str, ti_def_id, float(rsi_value))
            rows.append(rsi_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO rsi_daily (equity_id, trade_date, ti_def_id, rsi) "
                "VALUES (%s, %s, %s, %s)"
                "ON DUPLICATE KEY UPDATE rsi = VALUES(rsi)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} RSI values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_stoch(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Stochastic Oscillator values and upsert them into `stoch_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "STOCH"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    k_length: int = params_json["k"]
    d_length: int = params_json["d"]
    smooth_k: int = 3
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving stochastic oscillator values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: k_length={k_length}, d_length={d_length}, smooth_k={smooth_k}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float]] = []

        print(f"  → Calculating Stochastic with k_length={k_length}, d_length={d_length}...")
        stoch: pd.DataFrame = df.ta.stoch(
            high="high",
            low="low",
            close="close",
            k=k_length,
            d=d_length,
            smooth_k=smooth_k,
        )

        # Stochastic needs a warm-up period before values become non-NaN.
        warmup_periods: int = k_length + d_length + smooth_k - 3  # stochastic requires warmup period

        #  Prepare rows
        col_k: str = _get_pandas_ta_col_name(stoch, "STOCHk_")
        col_d: str = _get_pandas_ta_col_name(stoch, "STOCHd_")

        for date, row in stoch.iloc[warmup_periods:].iterrows():
            k_value = row[col_k]
            d_value = row[col_d]
            # Skip any remaining NaNs
            if pd.isna(k_value) or pd.isna(d_value):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            stoch_daily_row = (
                equity_id,
                date_str,
                ti_def_id,
                float(k_value),
                float(d_value),
            )
            rows.append(stoch_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO stoch_daily (equity_id, trade_date, ti_def_id, k, d) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE k = VALUES(k), d = VALUES(d)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} Stochastic values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_sma(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Simple Moving Average values and upsert them into `sma_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "SMA"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    sma_fast: int = params_json["sma_fast"]
    sma_slow: int = params_json["sma_slow"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving simple moving average values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: sma_fast={sma_fast}, sma_slow={sma_slow}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float]] = []

        print(f"  → Calculating Simple Moving Average with sma_fast={sma_fast}, sma_slow={sma_slow}...")
        sma_fast_series: pd.Series = df.ta.sma(
            close="close",
            length=sma_fast,
        )
        sma_slow_series: pd.Series = df.ta.sma(
            close="close",
            length=sma_slow,
        )
        sma_cross_df: pd.DataFrame = pd.concat(
            [
                sma_fast_series.rename("sma_fast"),
                sma_slow_series.rename("sma_slow"),
            ],
            axis=1,
        )
        if sma_cross_df.index.hasnans:
            raise ValueError("Issue concating fast and slow SMAs. Resulted in Nan index values")

        # SMA needs a warm-up period before values become non-NaN.
        warmup_periods: int = sma_slow - 1

        #  Prepare rows
        for date, row in sma_cross_df.iloc[warmup_periods:].iterrows():
            fast_sma = row["sma_fast"]
            slow_sma = row["sma_slow"]
            # Skip any remaining NaNs
            if pd.isna(fast_sma) or pd.isna(slow_sma):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            sma_daily_row = (
                equity_id,
                date_str,
                ti_def_id,
                float(fast_sma),
                float(slow_sma),
            )
            rows.append(sma_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO sma_daily (equity_id, trade_date, ti_def_id, sma_fast, sma_slow) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE sma_fast = VALUES(sma_fast), sma_slow = VALUES(sma_slow)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} SMA values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_ema(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Simple Moving Average values and upsert them into `ema_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "EMA"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    ema_fast: int = params_json["ema_fast"]
    ema_slow: int = params_json["ema_slow"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving simple moving average values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: ema_fast={ema_fast}, ema_slow={ema_slow}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float]] = []

        print(f"  → Calculating Exponential Moving Average with ema_fast={ema_fast}, ema_slow={ema_slow}...")
        ema_fast_series: pd.Series = df.ta.ema(
            close="close",
            length=ema_fast,
        )
        ema_slow_series: pd.Series = df.ta.ema(
            close="close",
            length=ema_slow,
        )
        ema_cross_df: pd.DataFrame = pd.concat(
            [
                ema_fast_series.rename("ema_fast"),
                ema_slow_series.rename("ema_slow"),
            ],
            axis=1,
        )
        if ema_cross_df.index.hasnans:
            raise ValueError("Issue concating fast and slow emas. Resulted in Nan index values")

        # ema needs a warm-up period before values become non-NaN.
        warmup_periods: int = ema_slow - 1

        #  Prepare rows
        for date, row in ema_cross_df.iloc[warmup_periods:].iterrows():
            fast_ema = row["ema_fast"]
            slow_ema = row["ema_slow"]
            # Skip any remaining NaNs
            if pd.isna(fast_ema) or pd.isna(slow_ema):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            ema_daily_row = (
                equity_id,
                date_str,
                ti_def_id,
                float(fast_ema),
                float(slow_ema),
            )
            rows.append(ema_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO ema_daily (equity_id, trade_date, ti_def_id, ema_fast, ema_slow) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE ema_fast = VALUES(ema_fast), ema_slow = VALUES(ema_slow)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} ema values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_adx(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Simple Moving Average values and upsert them into `ema_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "ADX"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    length: int = params_json["length"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving average directional index values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: length={length}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float, float]] = []

        print(f"  → Calculating ADX with length={length}...")
        adx_series: pd.DataFrame = df.ta.adx(
            high="high",
            low="low",
            close="close",
            length=length,
        )

        # adx needs a warm-up period before values become non-NaN.
        warmup_periods: int = length - 1

        #  Prepare rows
        col_adx: str = _get_pandas_ta_col_name(adx_series, "ADX_")
        col_plus_di: str = _get_pandas_ta_col_name(adx_series, "DMP_")
        col_minus_di: str = _get_pandas_ta_col_name(adx_series, "DMN_")

        for date, row in adx_series.iloc[warmup_periods:].iterrows():
            adx = row[col_adx]
            plus_di = row[col_plus_di]
            minus_di = row[col_minus_di]
            # Skip any remaining NaNs
            if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            adx_daily_row = (equity_id, date_str, ti_def_id, float(adx), float(plus_di), float(minus_di))
            rows.append(adx_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO adx_daily (equity_id, trade_date, ti_def_id, adx, plus_di, minus_di) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE adx = VALUES(adx), plus_di = VALUES(plus_di), minus_di = VALUES(minus_di)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} adx values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_bbands(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Bollinger Bands and upsert them into `bbands_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "BBANDS"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    std: int = params_json["std"]
    length: int = params_json["length"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving bollinger bands values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: std={std}, length={length}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float, float, float, float]] = []

        print(f"  → Calculating Bollinger bands with length={length}, std={std}...")
        bbands_series: pd.DataFrame = df.ta.bbands(
            close="close",
            length=length,
            std=std,
        )

        # BBANDS needs a warm-up period before values become non-NaN.
        warmup_periods: int = length - 1

        #  Prepare rows
        col_lower: str = _get_pandas_ta_col_name(bbands_series, "BBL_")
        col_middle: str = _get_pandas_ta_col_name(bbands_series, "BBM_")
        col_upper: str = _get_pandas_ta_col_name(bbands_series, "BBU_")
        col_bandwidth: str = _get_pandas_ta_col_name(bbands_series, "BBB_")
        col_percent_b: str = _get_pandas_ta_col_name(bbands_series, "BBP_")

        for date, row in bbands_series.iloc[warmup_periods:].iterrows():
            lower = row[col_lower]
            middle = row[col_middle]
            upper = row[col_upper]
            bandwidth = row[col_bandwidth]
            percent_b = row[col_percent_b]
            # Skip any remaining NaNs
            if pd.isna(lower) or pd.isna(middle) or pd.isna(upper) or pd.isna(bandwidth) or pd.isna(percent_b):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            bbands_daily_row = (equity_id, date_str, ti_def_id, float(upper), float(middle), float(lower), float(bandwidth), float(percent_b))
            rows.append(bbands_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO bbands_daily (equity_id, trade_date, ti_def_id, upper, middle, lower, bb_width, bb_percent_b) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE upper = VALUES(upper), middle = VALUES(middle), lower = VALUES(lower), bb_width = VALUES(bb_width), bb_percent_b = VALUES(bb_percent_b)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} bollinger bands values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_atr(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive Average True Range and upsert them into `atr_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "ATR"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    length: int = params_json["length"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving atr values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float]] = []

        print(f"  → Calculating Bollinger bands with length={length}...")
        atr_series: pd.Series = df.ta.atr(
            high="high",
            low="low",
            close="close",
            length=length,
        )

        # atr needs a warm-up period before values become non-NaN.
        warmup_periods: int = length - 1

        #  Prepare rows
        for date, atr_val in atr_series.iloc[warmup_periods:].items():
            # Skip any remaining NaNs
            if pd.isna(atr_val):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            atr_daily_row = (equity_id, date_str, ti_def_id, float(atr_val))
            rows.append(atr_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO atr_daily (equity_id, trade_date, ti_def_id, atr) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE atr = VALUES(atr)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} atr values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_obv(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive On-Balance Volume values and upsert them into `obv_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "OBV"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    signal: int = params_json["signal"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving on-balance volume values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: signal={signal}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, int]] = []

        print("  → Calculating OBV...")
        obv_series: pd.Series = df.ta.obv(close="close", volume="volume")

        # OBV is cumulative; use signal length as warm-up to align with planned smoothing.
        warmup_periods: int = max(signal - 1, 0)

        #  Prepare rows
        for date, obv_value in obv_series.iloc[warmup_periods:].items():
            # Skip any remaining NaNs
            if pd.isna(obv_value):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            obv_daily_row = (equity_id, date_str, ti_def_id, int(obv_value))
            rows.append(obv_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO obv_daily (equity_id, trade_date, ti_def_id, obv) "
                "VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE obv = VALUES(obv)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} obv values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_vol_roc(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive volume activity values and upsert them into `volume_activity_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "VOL_ROC"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    length: int = params_json["length"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving volume activity values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: length={length}")

        #  Calculate indicator (pandas)
        rows: list[tuple[int, str, int, float, float]] = []

        print(f"  → Calculating volume SMA and ratio with length={length}...")
        volume_sma: pd.Series = df["volume"].rolling(window=length).mean()
        volume_ratio: pd.Series = df["volume"] / volume_sma
        volume_activity_df: pd.DataFrame = pd.concat(
            [
                volume_sma.rename("volume_sma"),
                volume_ratio.rename("volume_ratio"),
            ],
            axis=1,
        )

        # Needs a warm-up period before values become non-NaN.
        warmup_periods: int = length - 1

        #  Prepare rows
        for date, row in volume_activity_df.iloc[warmup_periods:].iterrows():
            vol_sma = row["volume_sma"]
            vol_ratio = row["volume_ratio"]
            # Skip any remaining NaNs
            if pd.isna(vol_sma) or pd.isna(vol_ratio):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            volume_activity_row = (
                equity_id,
                date_str,
                ti_def_id,
                float(vol_sma),
                float(vol_ratio),
            )
            rows.append(volume_activity_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO volume_activity_daily (equity_id, trade_date, ti_def_id, volume_sma, volume_ratio) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE volume_sma = VALUES(volume_sma), volume_ratio = VALUES(volume_ratio)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} volume activity values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


def derive_macd(
    data: dict[str, pd.DataFrame],
    timeframe: int,
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
    equity_info: dict[str, int],
) -> None:
    """
    Derive MACD values and upsert them into `macd_daily`.

    `timeframe` selects the parameter set from `ti_definitions` (e.g. 3-day).
    """
    cursor: MySQLCursorAbstract = conn.cursor()
    ti_name: str = "MACD"
    ti_def_id, params_json = _get_ti_definition(cursor, ti_name, timeframe)
    fast: int = params_json["fast"]
    slow: int = params_json["slow"]
    signal: int = params_json["signal"]
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving MACD values")
        print("=" * 60)

        #  Get equity_id
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: fast={fast}, slow={slow}, signal={signal}")

        #  Calculate indicator (pandas_ta)
        rows: list[tuple[int, str, int, float, float, float]] = []

        print(f"  → Calculating MACD with fast={fast}, slow={slow}, signal={signal}...")
        macd_df: pd.DataFrame = df.ta.macd(
            close="close",
            fast=fast,
            slow=slow,
            signal=signal,
        )

        # MACD needs a warm-up period before values become non-NaN.
        warmup_periods: int = slow + signal - 2

        #  Prepare rows
        col_macd: str = _get_pandas_ta_col_name(macd_df, "MACD_")
        col_signal: str = _get_pandas_ta_col_name(macd_df, "MACDs_")
        col_hist: str = _get_pandas_ta_col_name(macd_df, "MACDh_")

        for date, row in macd_df.iloc[warmup_periods:].iterrows():
            macd_value = row[col_macd]
            signal_value = row[col_signal]
            hist_value = row[col_hist]
            # Skip any remaining NaNs
            if pd.isna(macd_value) or pd.isna(signal_value) or pd.isna(hist_value):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            macd_daily_row = (
                equity_id,
                date_str,
                ti_def_id,
                float(macd_value),
                float(signal_value),
                float(hist_value),
            )
            rows.append(macd_daily_row)

        #  Upsert into DB
        print(f"  → Prepared {len(rows)} rows for insertion")
        if rows:
            cursor.executemany(
                "INSERT INTO macd_daily (equity_id, trade_date, ti_def_id, macd, signal_line, hist) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE macd = VALUES(macd), signal_line = VALUES(signal_line), hist = VALUES(hist)",
                rows,
            )
        conn.commit()
        print(f"  ✓ Successfully upserted {len(rows)} macd values for {name} at {timeframe}-day timeframe")
        print("=" * 60)


if __name__ == "__main__":
    db_conn = get_db_conn()
    stock_data = get_stock_data()
    equity_info = get_equity_id()

    for tf in [3, 10, 20]:
        derive_rsi(stock_data, tf, db_conn, equity_info)
        derive_stoch(stock_data, tf, db_conn, equity_info)
        derive_sma(stock_data, tf, db_conn, equity_info)
        derive_ema(stock_data, tf, db_conn, equity_info)
        derive_adx(stock_data, tf, db_conn, equity_info)
        derive_bbands(stock_data, tf, db_conn, equity_info)
        derive_atr(stock_data, tf, db_conn, equity_info)
        derive_obv(stock_data, tf, db_conn, equity_info)
        derive_vol_roc(stock_data, tf, db_conn, equity_info)
        derive_macd(stock_data, tf, db_conn, equity_info)
