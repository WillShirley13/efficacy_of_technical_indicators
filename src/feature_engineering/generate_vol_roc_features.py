import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class VolRocFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        vol_current = self.raw_data["volume"]

        self.features["volume_current"] = vol_current
        vol_sma = vol_current.rolling(self.get_lookback_periods()["rolling_avg"], min_periods=1).mean()
        self.features["volume_sma"] = vol_sma
        vol_ratio = vol_current / vol_sma
        self.features["volume_ratio"] = vol_ratio
        self.features["volume_spike_mid"] = self.bin_threshold(vol_ratio, 1.5)
        self.features["volume_spike_high"] = self.bin_threshold(vol_ratio, 2)
        self.features["volume_trend"] = self.lin_reg_slope(vol_current, self.get_lookback_periods()["regression"])

        return self.features
