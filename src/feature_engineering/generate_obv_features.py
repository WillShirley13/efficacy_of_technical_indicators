import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class ObvFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        current = self.raw_data["obv"]
        close = self.raw_data["close"]

        self.features["OBV_current"] = current
        obv_sma = current.rolling(self.get_lookback_periods()["rolling_avg"], min_periods=1).mean()
        self.features["OBV_SMA"] = obv_sma
        self.features["OBV_vs_SMA"] = self.normalised_dist(current, obv_sma)
        self.features["OBV_delta"] = self.delta(current, current.shift(self.get_lookback_periods()["lag_long"]))
        self.features["OBV_trend"] = self.lin_reg_slope(current, self.get_lookback_periods()["regression"])

        obv_slope = self.lin_reg_slope(current, self.get_lookback_periods()["regression"])
        price_slope = self.lin_reg_slope(close, self.get_lookback_periods()["regression"])
        self.features["price_OBV_div_bull"] = self.bullish_divergence(obv_slope, price_slope)
        self.features["price_OBV_div_bear"] = self.bearish_divergence(obv_slope, price_slope)
        self.features["price_OBV_corrleated"] = self.obv_price_corrlated(obv_slope, price_slope)
        volume = self.raw_data["volume"]
        volume_ma = volume.rolling(self.get_lookback_periods()["rolling_avg"], min_periods=1).mean()
        self.features["volume_spike"] = self.volume_spike(volume, volume_ma)

        return self.features

    @staticmethod
    def bullish_divergence(obv_slope: pd.Series, price_slope: pd.Series) -> pd.Series:
        """
        Returns a binary series. If obv is up and price is down in the last
        """
        return (price_slope < 0) & (obv_slope > 0)

    @staticmethod
    def bearish_divergence(obv_slope: pd.Series, price_slope: pd.Series) -> pd.Series:
        """
        Returns a binary series. If obv is up and price is down in the last
        """
        return (price_slope > 0) & (obv_slope < 0)

    @staticmethod
    def obv_price_corrlated(obv_slope: pd.Series, price_slope: pd.Series) -> pd.Series:
        """
        Returns a binary series. If obv is up and price is down in the last
        """
        return ((price_slope > 0) & (obv_slope > 0)) | ((price_slope < 0) & (obv_slope < 0))

    @staticmethod
    def volume_spike(volume: pd.Series, volume_ma: pd.Series) -> pd.Series:
        """
        Returns a binary series. If volume is more than 1.5 times the MA, return 1, otherwise 0.
        """
        return (volume > volume_ma * 1.5).astype(int)
