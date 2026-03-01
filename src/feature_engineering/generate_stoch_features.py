import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class StochFeatureGenerator(BaseFeatureGenerator):
    """
    Stochastic Oscillator feature derivation class.
    Expects raw_data to contain at minimum 'k' and 'd' columns.
    NaNs are NOT dropped in any method — drop once at the end after all features are built.
    """

    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        k = self.raw_data["k"]
        d = self.raw_data["d"]

        lookbacks = self.get_lookback_periods()
        self.features["STOCH_K"] = k
        self.features["STOCH_D"] = d
        self.features["STOCH_K_lag"] = self.lag_n(k, lag=lookbacks["lag_short"])
        self.features["STOCH_D_lag"] = self.lag_n(d, lag=lookbacks["lag_short"])
        self.features["STOCH_crossover"] = self.crossover(k, d, lookback=lookbacks["crossover"])
        self.features["STOCH_K_delta"] = self.delta(k, self.features["STOCH_K_lag"])
        self.features["STOCH_dist_20"] = self.dist_from_constant(k, 20).abs()
        self.features["STOCH_dist_80"] = self.dist_from_constant(k, 80).abs()
        self.features["STOCH_in_oversold"] = self.bin_threshold(k, 20, above_threshold=False)
        self.features["STOCH_in_overbought"] = self.bin_threshold(k, 80, above_threshold=True)

        return self.features
