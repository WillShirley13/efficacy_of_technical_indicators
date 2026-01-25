import json
from mysql.connector.abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
from mysql.connector.pooling import PooledMySQLConnection
import pandas as pd
import pandas_ta  # noqa: F401  (registers the `.ta` accessor on pandas objects)
from pathlib import Path
from typing import Final, cast
import mysql.connector
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
CLEAN_DATA_DIR: Final[Path] = BASE_DIR / "data/clean"


def get_db_conn() -> PooledMySQLConnection | MySQLConnectionAbstract:
    # Load DB credentials from `.env` (keeps secrets out of the repo).
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
        # Read the cleaned CSV for this ticker. `date` is both parsed and used as the index.
        stock_data[stock] = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
            f"{CLEAN_DATA_DIR}/{stock}.csv", index_col="date", parse_dates=["date"]
        )

        if stock_data[stock].empty:
            raise ValueError(f"Failed to load {stock}'s data")

        # Validate expected columns early so the indicator calcs don't silently produce garbage.
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

        print(f"{stock_data[stock].info()}\n")

    return stock_data


def get_equity_id(
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
) -> dict[str, int]:
    """
    Fetch `{ticker: equity_id}` from the `equities` table.
    """
    expected_tickers: set[str] = {"EEM", "EFA", "IWM", "QQQ", "SPY"}
    cursor: MySQLCursorAbstract = conn.cursor()

    # Pull all equity IDs up-front so per-indicator loops don't keep hitting the DB.
    cursor.execute("SELECT equity_id, ticker FROM equities ORDER BY ticker")
    result = cursor.fetchall()
    if not result:
        raise ValueError("Unable to retrieve equity_ids and tickers from equities table")

    # Convert DB rows into a `{ticker: id}` mapping.
    equity_info: dict[str, int] = {cast(str, ticker): cast(int, equity_id) for equity_id, ticker in result}

    # Sanity check: expected tickers are a fixed set of ETFs in the DB.
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

        # --- Prepare rows (Python) ---
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
                "VALUES (%s, %s, %s, %s) "
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

        # --- Prepare rows (Python) ---
        for date, k_value, d_value in stoch.iloc[warmup_periods:].itertuples(index=True, name=None):
            # Skip any remaining NaNs (e.g. missing OHLC rows) to avoid DB errors / CHECK failures.
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


if __name__ == "__main__":
    # --- Run ---
    db_conn = get_db_conn()
    stock_data = get_stock_data()
    equity_info = get_equity_id(db_conn)

    # derive_rsi(stock_data, 3, db_conn, equity_info)
    derive_stoch(stock_data, 3, db_conn, equity_info)
