from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR: Final[Path] = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR: Final[Path] = BASE_DIR / "data" / "clean"


def get_stock_data() -> dict[str, pd.DataFrame]:
    """
    Load and return all stock data as a dict
    """
    stock_data: dict[str, pd.DataFrame] = {
        "EEM": pd.DataFrame(),
        "EFA": pd.DataFrame(),
        "IWM": pd.DataFrame(),
        "QQQ": pd.DataFrame(),
        "SPY": pd.DataFrame(),
    }

    for stock in stock_data.keys():
        stock_data[stock] = pd.read_csv(f"{RAW_DATA_DIR}/{stock}.csv")  # pyright: ignore[reportUnknownMemberType]

        if stock_data[stock].empty:
            raise ValueError(f"Failed to load {stock}'s data")

        required_cols: set[str] = {
            "date",
            "close",
            "high",
            "low",
            "open",
            "volume",
        }
        actual_cols: set[str] = set(stock_data[stock].columns)
        if required_cols - actual_cols:
            raise ValueError(f"{stock} Missing required columns, has {actual_cols}, needs {required_cols}")

        print(f"{stock_data[stock].info()}\n")

    return stock_data


def logical_checks(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Perform logical check to ensure reliability of data
    Checks:
    - Validate high of day is greater than or equal to low of day
    - Validate price is always non-negative
    - Validate volume is non-negative and greater than zero
    - Duplicate rows
    """
    cleaned: dict[str, pd.DataFrame] = {}

    for stock_name, data in stock_data.items():
        df: pd.DataFrame = data.copy()

        # If Date exists, use it as the index (duplicate-date checks become meaningful)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        bad_date: pd.Series = df["date"].isna()
        if bad_date.any():
            print(f"\n[{stock_name}] Bad Date rows: {int(bad_date.sum())}")
            print(f"No of bad rows: {df.loc[bad_date].shape}")
        df = df.loc[~bad_date].set_index("date")
        df = df.sort_index()

        # Ensure all numerical values remain numerical after reading in the csv files
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(inplace=True)

        print(f"\n[{stock_name}] Before cleaning: rows={len(df)}, cols={df.shape[1]}")

        # Validate high of day is greater than or equal to low of day.
        high_lt_low: pd.Series = df["high"] < df["low"]
        if high_lt_low.any():
            print(f"[{stock_name}] High < Low violations: {int(high_lt_low.sum())}")
            print(df.loc[high_lt_low, ["open", "high", "low", "close", "volume"]].head(5))

        # Validate prices are non-negative
        price_lt_zero: pd.Series = (df[["close", "high", "low", "open"]] < 0).any(axis=1)
        if price_lt_zero.any():
            print(f"[{stock_name}] Negative price violations: {int(price_lt_zero.sum())}")
            print(df.loc[price_lt_zero, ["open", "high", "low", "close", "volume"]].head(5))

        # Validate volume is larger than zero
        vol_le_zero: pd.Series = df["volume"] <= 0
        if vol_le_zero.any():
            print(f"[{stock_name}] Volume <= 0 violations: {int(vol_le_zero.sum())}")
            print(df.loc[vol_le_zero, ["open", "high", "low", "close", "volume"]].head(5))

        # Drop rows violating assertions
        drop_mask: pd.Series = high_lt_low | price_lt_zero | vol_le_zero
        if drop_mask.any():
            df = df.loc[~drop_mask].copy()
            print(f"[{stock_name}] Dropped assertion-violating rows: {int(drop_mask.sum())}")

        # Duplicate dates — keep the last occurrence
        dup_idx_all: NDArray[np.bool_] = df.index.duplicated(keep=False)
        if dup_idx_all.any():
            dup_idx_count: int = int(dup_idx_all.sum())
            print(f"[{stock_name}] Duplicate index (date) rows: {dup_idx_count} (keeping last)")
            print(df.loc[dup_idx_all].head(5))
            dup_idx_drop: NDArray[np.bool_] = df.index.duplicated(keep="last")
            df = df.loc[~dup_idx_drop].copy()

        # Duplicate full rows — keep the last occurrence
        dup_row_all: pd.Series = df.duplicated(keep=False)
        if dup_row_all.any():
            dup_row_count: int = int(dup_row_all.sum())
            print(f"[{stock_name}] Duplicate full rows: {dup_row_count} (keeping last)")
            print(df.loc[dup_row_all].head(5))
            dup_row_drop: pd.Series = df.duplicated(keep="last")
            df = df.loc[~dup_row_drop].copy()

        print(f"[{stock_name}] After cleaning: rows={len(df)}, cols={df.shape[1]}")
        print(df.head(5))

        cleaned[stock_name] = df

    return cleaned


def save_data(stock_data: dict[str, pd.DataFrame]) -> None:
    for stock, df in stock_data.items():
        out_path: Path = CLEAN_DATA_DIR / f"{stock}.csv"
        df.to_csv(out_path, index=True, index_label="date")
        print(f"{stock} cleaned data saved as CSV to {out_path}")


raw_data: dict[str, pd.DataFrame] = get_stock_data()
cleaned_data: dict[str, pd.DataFrame] = logical_checks(raw_data)
save_data(cleaned_data)
