import pandas as pd


# Returns Series lagged by 'lag' periods (i.e. shifted down), with NaNs dropped.
def lag_n(series: pd.Series, lag: int) -> pd.Series:
    lag_n = series.shift(lag)
    lag_n.dropna(inplace=True)
    return lag_n


# Calculates the difference between two Series, element-wise
def delta(series1: pd.Series, series2: pd.Series) -> pd.Series:
    return series1 - series2


# Computes the difference between each element in the Series and a given constant.
def dist_from_constant(series: pd.Series, constant: int) -> pd.Series:
    return series - constant


# Returns a binary Series indicating whether each value is above or below a threshold.
# If above_threshold is True, returns 1 for values above the threshold, 0 otherwise.
# If above_threshold is False, returns 1 for values below the threshold, 0 otherwise.
def bin_threshold(series: pd.Series, threshold: int, above_threshold=True) -> pd.Series:
    if above_threshold:
        return (series > threshold).astype(int)
    return (series < threshold).astype(int)


# Return a binary series indicating if series1 crossed over (s1 > s2) series2 at anypoint in the last n rows
def crossover(series1: pd.Series, series2: pd.Series, lookback: int) -> pd.Series:
    all_croses = (series1 > series2).astype(int)
    return all_croses.shift(1).rolling(window=lookback, min_periods=1).max().dropna()
