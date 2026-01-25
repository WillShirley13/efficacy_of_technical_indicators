import yfinance as yf  # type: ignore[import-untyped]
import pandas as pd
from pathlib import Path


START_DATE = "2006-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"
TICKERS: list[str] = ["SPY", "QQQ", "IWM", "EFA", "EEM"]

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data/raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def print_df_summary(df: pd.DataFrame, *, name: str, head: int = 5) -> None:
    print(f"\n--- DataFrame Summary for {name} ---")
    print(f"Shape: {df.shape} (rows, columns)")
    if not df.empty:
        print(f"Date range: {df.index.min()} -> {df.index.max()}")
    print(f"\nFirst {head} rows:")
    print(df.head(head))

    # `info()` already includes columns, non-null counts, and dtypes, so don't print those separately.
    print("\nColumn info (non-null counts + dtypes):")
    df.info()

    print("\nDescriptive statistics (numerical columns):")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isna().sum())
    print("--- End of DataFrame Summary ---\n")


for ticker in TICKERS:
    df: pd.DataFrame = yf.download(  # type: ignore
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        progress=False,
        auto_adjust=True,
    )
    df.columns = [col.lower() for col in df.columns.get_level_values(0).tolist()]
    if df.empty:
        print(f"Failed to download data for {ticker}")
        continue
    print_df_summary(df, name=ticker)
    df.to_csv(OUT_DIR / f"{ticker}.csv", index=True, index_label="date")
