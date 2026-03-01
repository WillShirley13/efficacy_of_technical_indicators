import pandas as pd

from src.feature_engineering.common import BaseFeatureGenerator


class MacdFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        super().__init__(raw_data, timeframe)

    def generate(self) -> pd.DataFrame:
        macd = self.raw_data["macd"]
        signal_line = self.raw_data["signal_line"]
        hist = self.raw_data["hist"]

        self.features["MACD_line"] = macd
        self.features["signal_line"] = signal_line
        self.features["hist"] = hist
        lookback_lag = self.get_lookback_periods()["lag_long"]
        self.features["MACD_line_lag"] = self.lag_n(macd, lookback_lag)
        self.features["MACD_signal_lag"] = self.lag_n(signal_line, lookback_lag)
        macd_hist_lag = self.lag_n(hist, lookback_lag)
        self.features["MACD_histogram_lag"] = macd_hist_lag
        lookback_crossover = self.get_lookback_periods()["crossover"]
        self.features["MACD_cross_above"] = self.crossover(macd, signal_line, lookback_crossover)
        self.features["MACD_cross_below"] = self.crossover(signal_line, macd, lookback_crossover)
        self.features["MACD_hist_delta"] = self.delta(hist, macd_hist_lag)
        self.features["MACD_bullish"] = self.macd_bullish(macd, signal_line)
        centreline_up = macd.shift(1).rolling(lookback_crossover, min_periods=1).apply(lambda win: (win > 0).any())
        centreline_down = macd.shift(1).rolling(lookback_crossover, min_periods=1).apply(lambda win: (win < 0).any())
        self.features["MACD_centreline_cross_up"] = centreline_up.fillna(0).astype(int)
        self.features["MACD_centreline_cross_down"] = centreline_down.fillna(0).astype(int)
        self.features["MACD_above_zero"] = self.bin_threshold(macd, 0)

        return self.features

    @staticmethod
    def macd_bullish(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
        return ((macd_line > signal_line) & (macd_line > 0)).astype(int)
