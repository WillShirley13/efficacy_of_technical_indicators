import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class AtrFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        current = self.raw_data["atr"]
        close = self.raw_data["close"]

        lookbacks = self.get_lookback_periods()
        self.features["ATR_current"] = current
        self.features["ATR_normalised"] = current / close
        atr_lag = self.lag_n(current, lookbacks["lag_long"])
        self.features["ATR_lag"] = atr_lag
        atr_delta = self.delta(current, atr_lag)
        self.features["ATR_delta"] = atr_delta
        atr_percentile = current.rolling(lookbacks["percentile"], min_periods=1).rank(pct=True)
        self.features["ATR_percentile"] = atr_percentile
        self.features["ATR_expanding"] = self.bin_threshold(atr_delta, threshold=0, above_threshold=True)
        self.features["ATR_contracting"] = self.bin_threshold(atr_delta, threshold=0, above_threshold=False)
        self.features["ATR_high"] = self.bin_threshold((atr_percentile * 100).astype(int), threshold=75, above_threshold=True)
        self.features["ATR_low"] = self.bin_threshold((atr_percentile * 100).astype(int), threshold=25, above_threshold=False)

        return self.features
