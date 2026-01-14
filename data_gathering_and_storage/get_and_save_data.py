import yfinance as yf  # type: ignore[import-untyped]
import pandas as pd
import pandas_ta as ta

df: pd.DataFrame | None = yf.download("SPY", start="2010-01-01", end="2025-12-31")
if df is None:
    raise ValueError("No data found")


close = df["Close"]
if isinstance(close, pd.DataFrame):  # MultiIndex columns case from yfinance
    close = close.iloc[:, 0]  # pick SPY

rsi = ta.rsi(close, length=14)
print(rsi.head(20))
print(rsi.tail())
