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
    def bin_threshold(series: pd.Series, threshold: int, above_threshold: bool = True) -> pd.Series:
        """
        Returns a binary Series indicating whether each value is above or below a threshold.
        If above_threshold is True, returns 1 for values above the threshold, 0 otherwise.
        If above_threshold is False, returns 1 for values below the threshold, 0 otherwise.
        """
        if above_threshold:
            return (series > threshold).astype(int)
        return (series < threshold).astype(int)

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
    def lin_reg_slope(series: pd.Series, timeframe: int) -> pd.Series:
        """
        Returns the gradient/slope of the linear regression line of the last n_days of data.
        E.g. Steeper the gradient, the faster the recent change.
        """
        lookback: int
        match timeframe:
            case 3:
                lookback = 5
            case 10:
                lookback = 15
            case 20:
                lookback = 25
            case _:
                raise ValueError("Timeframe must be 3, 10 or 20 (days)")

        return series.rolling(window=lookback).apply(
            lambda window: np.polyfit(range(len(window)), window, deg=1)[0],
            raw=True,
        )
