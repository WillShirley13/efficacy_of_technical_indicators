import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class RsiFeatureGenerator(BaseFeatureGenerator):
    """
    RSI feature derivation class.
    Expects raw_data to contain at minimum a 'rsi' column.
    NaNs are NOT dropped in any method — drop once at the end after all features are built.
    """

    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        rsi = self.raw_data["rsi"]

        lookbacks = self.get_lookback_periods()
        self.features["RSI_current"] = rsi
        self.features["RSI_lag_short"] = self.lag_n(rsi, lag=lookbacks["lag_short"])
        self.features["RSI_lag_long"] = self.lag_n(rsi, lag=lookbacks["lag_long"])
        self.features["RSI_delta_short"] = self.delta(rsi, self.features["RSI_lag_short"])
        self.features["RSI_delta_long"] = self.delta(rsi, self.features["RSI_lag_long"])
        self.features["RSI_dist_30"] = self.dist_from_constant(rsi, 30)
        self.features["RSI_dist_70"] = self.dist_from_constant(rsi, 70)

        return self.features
