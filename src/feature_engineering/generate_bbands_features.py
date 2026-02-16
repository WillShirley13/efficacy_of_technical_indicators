import pandas as pd

from feature_engineering.common import BaseFeatureGenerator


class BBandsFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        upper = self.raw_data["upper"]
        lower = self.raw_data["lower"]
        middle = self.raw_data["middle"]
        bb_width = self.raw_data["bb_width"]
        bb_percent_b = self.raw_data["bb_percent_b"]

        self.features["BB_upper"] = upper
        self.features["BB_lower"] = lower
        self.features["BB_middle"] = middle
        self.features["BB_width"] = bb_width
