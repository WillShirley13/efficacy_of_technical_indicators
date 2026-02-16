import os
from pathlib import Path
from typing import Final, cast

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from mysql.connector.abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
from mysql.connector.pooling import PooledMySQLConnection

# Paths / constants
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
CLEAN_DATA_DIR: Final[Path] = BASE_DIR / "data" / "clean"


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
    - Each DataFrame is indexed by `date`.
    - Column names are standardised to lower-case - open/high/low/close/volume.
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
        stock_data[stock] = pd.read_csv(f"{CLEAN_DATA_DIR}/{stock}.csv", index_col="date", parse_dates=["date"])

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

        # print(f"{stock_data[stock].info}\n")

    return stock_data


def get_equity_id() -> dict[str, int]:
    """
    Fetch `{ticker: equity_id}` from the `equities` table.
    """
    conn = get_db_conn()
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
