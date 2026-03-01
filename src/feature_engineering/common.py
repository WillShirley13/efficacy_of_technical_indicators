import numpy as np
import pandas as pd


class BaseFeatureGenerator:
    """
    Base class providing reusable feature transformation methods.
    TI-specific feature generators inherit from this and implement
    their own derivation logic via the generate() method.

    Attributes:
        raw_data: The raw indicator DataFrame from the DB.
        timeframe: Prediction horizon (3, 10, or 20 days).
        features: DataFrame holding all engineered features. Populated by generate().
    """

    def __init__(self, raw_data: pd.DataFrame, timeframe: int) -> None:
        self.raw_data = raw_data
        self.timeframe = timeframe
        self.features: pd.DataFrame = pd.DataFrame()

    # Timeframe → scaled lookback periods.
    # Keys: rolling_avg, percentile, regression, crossover, lag_short, lag_long
    _LOOKBACK_MAP: dict[int, dict[str, int]] = {
        3: {"rolling_avg": 10, "percentile": 30, "regression": 5, "crossover": 2, "lag_short": 3, "lag_long": 5},
        10: {"rolling_avg": 20, "percentile": 60, "regression": 15, "crossover": 3, "lag_short": 5, "lag_long": 10},
        20: {"rolling_avg": 40, "percentile": 120, "regression": 25, "crossover": 4, "lag_short": 8, "lag_long": 15},
    }

    def get_lookback_periods(self) -> dict[str, int]:
        """
        Returns a dict of lookback periods scaled to self.timeframe.
        """
        return self._LOOKBACK_MAP[self.timeframe]

    def generate(self) -> pd.DataFrame:
        """
        Engineer all features for this TI. Subclasses must override this.
        Should populate self.features and return it.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @staticmethod
    def lag_n(series: pd.Series, lag: int) -> pd.Series:
        """Returns the value at index n-lag for each element_n."""
        return series.shift(lag)

    @staticmethod
    def delta(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Calculates the difference between two Series, element-wise."""
        return series1 - series2

    @staticmethod
    def dist_from_constant(series: pd.Series, constant: int) -> pd.Series:
        """Computes the difference between each element in the Series and a given constant."""
        return series - constant

    @staticmethod
    def bin_threshold(series: pd.Series, threshold: int | float, above_threshold: bool = True) -> pd.Series:
        """
        Returns a binary Series indicating whether each value is above or below a threshold.
        If above_threshold is True, returns 1 for values above the threshold, 0 otherwise.
        If above_threshold is False, returns 1 for values below the threshold, 0 otherwise.
        """
        if above_threshold:
            return (series > threshold).astype(int)
        return (series < threshold).astype(int)

    @staticmethod
    def bin_dynamic_threshold(series: pd.Series, d_threshold: pd.Series, above_threshold: bool = True) -> pd.Series:
        """
        Returns a binary Series indicating whether each value is above or below a threshold.
        If above_threshold is True, returns 1 for values above the threshold, 0 otherwise.
        If above_threshold is False, returns 1 for values below the threshold, 0 otherwise.
        """
        if above_threshold:
            return (series > d_threshold).astype(int)
        return (series < d_threshold).astype(int)

    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series, lookback: int) -> pd.Series:
        """
        Returns a binary series indicating if series1 crossed over (s1 > s2) series2
        at any point in the last n rows.
        """
        crossed = (series1 > series2).astype(int)
        return crossed.shift(1).rolling(window=lookback, min_periods=1).max()

    @staticmethod
    def normalised_dist(series: pd.Series, reference: pd.Series) -> pd.Series:
        """
        Returns the normalised distance of series from reference, element-wise.
        E.g. normalised_dist(sma_fast, sma_slow) = (sma_fast - sma_slow) / sma_slow
        Also used for price-vs-line: normalised_dist(close, sma_fast) = (close - sma_fast) / sma_fast
        """
        return (series - reference) / reference

    @staticmethod
    def lin_reg_slope(series: pd.Series, lookback: int) -> pd.Series:
        """
        Returns the gradient/slope of the linear regression line over the
        given lookback window. Steeper gradient → faster recent change.

        Use get_lookback_periods()["regression"] for the lookback value.
        """
        return series.rolling(window=lookback).apply(
            lambda window: np.polyfit(range(len(window)), window, deg=1)[0],
            raw=True,
        )
