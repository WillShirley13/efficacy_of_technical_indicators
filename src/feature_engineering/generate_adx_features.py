import pandas as pd

from feature_engineering.common import BaseFeatureGenerator


class AdxFeatureGenerator(BaseFeatureGenerator):
    """
    Average Directional Index feature derivation class.
    Expects raw_data to contain at minimum 'adx', 'plus_di', and 'minus_di' columns.
    NaNs are NOT dropped in any method — drop once at the end after all features are built.
    """

    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        adx = self.raw_data["adx"]
        plus_di = self.raw_data["plus_di"]
        minus_di = self.raw_data["minus_di"]

        self.features["ADX_current"] = adx
        adx_lag3 = self.lag_n(adx, 3)
        self.features["ADX_lag3"] = adx_lag3
        adx_lag5 = self.lag_n(adx, 5)
        self.features["ADX_lag5"] = adx_lag5
        self.features["ADX_delta"] = self.delta(adx, adx_lag5)
        self.features["ADX_above_20"] = self.bin_threshold(adx, 20)
        self.features["ADX_above_25"] = self.bin_threshold(adx, 25)
        self.features["ADX_above_40"] = self.bin_threshold(adx, 40)
        self.features["PLUS_DI"] = plus_di
        self.features["MINUS_DI"] = minus_di
        self.features["DI_DIFF"] = self.delta(plus_di, minus_di)
        self.features["DI_crossover"] = self.crossover(plus_di, minus_di, int(self.timeframe * 1.5))
        self.features["ADX_slope"] = self.lin_reg_slope(adx, self.timeframe)

        return self.features
