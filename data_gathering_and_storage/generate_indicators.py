import json
import os
from pathlib import Path
from typing import Final, cast

import mysql.connector
import pandas as pd
import pandas_ta  # noqa: F401
from dotenv import load_dotenv
from mysql.connector.abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
from mysql.connector.pooling import PooledMySQLConnection

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
CLEAN_DATA_DIR: Final[Path] = BASE_DIR / "data/clean"


def _get_pandas_ta_col_name(df: pd.DataFrame, prefix: str) -> str:
    """
    Pick exactly one column from `df` whose name starts with `prefix`.

    This avoids relying on pandas_ta column ordering (which can change) while still
    being robust to suffix formatting differences (e.g. `2` vs `2.0`).
    """
    matches: list[str] = [str(c) for c in df.columns if str(c).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly 1 column starting with {prefix!r}, got {matches}")
    return matches[0]


def get_db_conn() -> PooledMySQLConnection | MySQLConnectionAbstract:
    # Load DB credentials from `.env`
    load_dotenv()

    # Create a single connection for the whole run.
    conn = mysql.connector.connect(
        host="localhost",
        user=os.getenv("STOCK_DATA_USER"),
        password=os.getenv("STOCK_DATA_PW"),
        database="stock_data",
    )
    print("--- Database connection successful ---")

    return conn


def get_stock_data() -> dict[str, pd.DataFrame]:
    """
    Load the cleaned OHLCV CSVs into a `{ticker: DataFrame}` dict.

    Notes:
    - Each DataFrame is indexed by `date` (parsed as datetime).
    - Column names are standardised to lower-case (`open/high/low/close/volume`).
    """
    stock_data: dict[str, pd.DataFrame] = {
        "EEM": pd.DataFrame(),
        "EFA": pd.DataFrame(),
        "IWM": pd.DataFrame(),
        "QQQ": pd.DataFrame(),
        "SPY": pd.DataFrame(),
    }

    for stock in stock_data.keys():
        # Read the cleaned CSV for this ticker. `date` is used as the index.
        stock_data[stock] = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            f"{CLEAN_DATA_DIR}/{stock}.csv", index_col="date", parse_dates=["date"]
        )

        if stock_data[stock].empty:
            raise ValueError(f"Failed to load {stock}'s data")

        # Validate expected OHLCV columns
        required_cols: set[str] = {
            "date",
            "close",
            "high",
            "low",
            "open",
            "volume",
        }
        actual_cols: set[str] = set(list(stock_data[stock].columns) + [str(stock_data[stock].index.name)])
        if required_cols - actual_cols:
            raise ValueError(f"{stock} Missing required columns, has {actual_cols}, needs {required_cols}")

        print(f"{stock_data[stock].info}\n")

    return stock_data


def get_equity_id(
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
) -> dict[str, int]:
    """
    Fetch `{ticker: equity_id}` from the `equities` table.
    """
    expected_tickers: set[str] = {"EEM", "EFA", "IWM", "QQQ", "SPY"}
    cursor: MySQLCursorAbstract = conn.cursor()

    # Get all equity IDs and tickers
    cursor.execute("SELECT equity_id, ticker FROM equities ORDER BY ticker")
    result = cursor.fetchall()
    if not result:
        raise ValueError("Unable to retrieve equity_ids and tickers from equities table")

    equity_info: dict[str, int] = {cast(str, ticker): cast(int, equity_id) for equity_id, ticker in result}

    # Check that all expected tickers are in the database
    if expected_tickers - set(equity_info.keys()):
        raise ValueError(f"Expected tickers {expected_tickers} not found in equities table")
    return equity_info


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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving relative strength index values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        length: int = params_json["length"]
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: length={length}")

        # --- Calculate indicator (pandas_ta) ---
        rows: list[tuple[int, str, int, float]] = []

        print(f"  → Calculating RSI with length={length}...")
        rsi: pd.Series = df.ta.rsi(close="close", length=length)

        # --- Prepare rows ---
        for date, rsi_value in rsi.iloc[length:].items():
            # Skip warm-up NaNs and any missing values (avoids MySQL insert issues).
            if pd.isna(rsi_value):
                continue
            date_ts: pd.Timestamp = cast(pd.Timestamp, date)
            date_str: str = date_ts.strftime("%Y-%m-%d")  # Convert to string for MySQL
            rsi_daily_row = (equity_id, date_str, ti_def_id, float(rsi_value))
            rows.append(rsi_daily_row)

        # --- Upsert into DB ---
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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving stochastic oscillator values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        k_length: int = params_json["k"]
        d_length: int = params_json["d"]
        smooth_k: int = 3
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: k_length={k_length}, d_length={d_length}, smooth_k={smooth_k}")

        # --- Calculate indicator (pandas_ta) ---
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

        # --- Prepare rows ---
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

        # --- Upsert into DB ---
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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving simple moving average values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        sma_fast: int = params_json["sma_fast"]
        sma_slow: int = params_json["sma_slow"]
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: sma_fast={sma_fast}, sma_slow={sma_slow}")

        # --- Calculate indicator (pandas_ta) ---
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

        # --- Prepare rows ---
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

        # --- Upsert into DB ---
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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving simple moving average values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        ema_fast: int = params_json["ema_fast"]
        ema_slow: int = params_json["ema_slow"]
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: ema_fast={ema_fast}, ema_slow={ema_slow}")

        # --- Calculate indicator (pandas_ta) ---
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

        # --- Prepare rows ---
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

        # --- Upsert into DB ---
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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving average directional index values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        length: int = params_json["length"]
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: length={length}")

        # --- Calculate indicator (pandas_ta) ---
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

        # --- Prepare rows ---
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

        # --- Upsert into DB ---
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
    for name, df in data.items():
        print("\n" + "=" * 60)
        print(f"Processing ETF: {name} - Deriving bollinger bands values")
        print("=" * 60)

        # --- Get equity_id ---
        equity_id: int = equity_info[name]
        print(f"  → Retrieved equity_id: {equity_id}")

        cursor.execute(
            "SELECT ti_def_id, params_json FROM ti_definitions where ti_name = %s AND time_frame = %s",
            (ti_name, timeframe),
        )
        result = cursor.fetchone()
        if result is None:
            raise ValueError(f"No TI definition found for {ti_name} with timeframe {timeframe}")
        ti_def_id: int = int(result[0])  # type: ignore[arg-type]
        params_json: dict[str, int] = json.loads(str(result[1]))  # type: ignore[arg-type]
        std: int = params_json["std"]
        length: int = params_json["length"]
        print(f"  → Retrieved ti_def_id: {ti_def_id}")
        print(f"  → Parameters: std={std}, length={length}")

        # --- Calculate indicator (pandas_ta) ---
        rows: list[tuple[int, str, int, float, float, float, float, float]] = []

        print(f"  → Calculating Bollinger bands with length={length}, std={std}...")
        bbands_series: pd.DataFrame = df.ta.bbands(
            close="close",
            length=length,
            std=std,
        )

        # BBANDS needs a warm-up period before values become non-NaN.
        warmup_periods: int = length - 1

        # --- Prepare rows ---
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

        # --- Upsert into DB ---
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


if __name__ == "__main__":
    db_conn = get_db_conn()
    stock_data = get_stock_data()
    equity_info = get_equity_id(db_conn)

    for tf in [3, 10, 20]:
        derive_rsi(stock_data, tf, db_conn, equity_info)
        derive_stoch(stock_data, tf, db_conn, equity_info)
        derive_sma(stock_data, tf, db_conn, equity_info)
        derive_ema(stock_data, tf, db_conn, equity_info)
        derive_adx(stock_data, tf, db_conn, equity_info)
        derive_bbands(stock_data, tf, db_conn, equity_info)
