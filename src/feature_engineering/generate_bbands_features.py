import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class BbandsFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        close = self.raw_data["close"]
        upper = self.raw_data["upper"]
        lower = self.raw_data["lower"]
        middle = self.raw_data["middle"]
        bb_width = self.raw_data["bb_width"]
        bb_percent_b = self.raw_data["bb_percent_b"]

        self.features["BB_upper"] = upper
        self.features["BB_lower"] = lower
        self.features["BB_middle"] = middle
        self.features["BB_width"] = bb_width
        self.features["BB_percent_b"] = bb_percent_b

        width_avg = bb_width.rolling(self.get_lookback_periods()["rolling_avg"], min_periods=1).mean()
        self.features["BB_squeeze"] = self.bin_dynamic_threshold(bb_width, width_avg, above_threshold=False)
        self.features["BB_expansion"] = self.bin_dynamic_threshold(bb_width, width_avg, above_threshold=True)
        self.features["price_gt_upper"] = self.bin_dynamic_threshold(close, upper)
        self.features["price_lt_lower"] = self.bin_dynamic_threshold(close, lower, above_threshold=False)

        return self.features
