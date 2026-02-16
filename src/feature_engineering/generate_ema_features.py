import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class EmaFeatureGenerator(BaseFeatureGenerator):
    """
    Simple Moving Average feature derivation class.
    Expects raw_data to contain at minimum 'ema_fast', 'ema_slow', and 'close' columns.
    NaNs are NOT dropped in any method — drop once at the end after all features are built.
    """

    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        fast = self.raw_data["ema_fast"]
        slow = self.raw_data["ema_slow"]
        close = self.raw_data["close"]

        self.features["EMA_fast"] = fast
        self.features["EMA_slow"] = slow
        self.features["EMA_distance"] = self.normalised_dist(fast, slow)
        self.features["EMA_cross_above"] = self.crossover(fast, slow, lookback=2)
        self.features["EMA_cross_below"] = self.crossover(slow, fast, lookback=2)
        self.features["EMA_bullish"] = self.bin_threshold(self.delta(fast, slow), threshold=0, above_threshold=True)
        self.features["Price_vs_fast"] = self.normalised_dist(close, fast)
        self.features["Price_vs_slow"] = self.normalised_dist(close, slow)
        self.features["EMA_distance_lag5"] = self.lag_n(self.features["EMA_distance"], lag=5)
        self.features["EMA_distance_delta"] = self.delta(self.features["EMA_distance"], self.features["EMA_distance_lag5"])

        return self.features
