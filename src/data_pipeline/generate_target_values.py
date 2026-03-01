import numpy as np
import pandas as pd

from src.utils import get_equity_id, get_stock_data

SIGMA_MULTIPLE = 1.25  # Multipler for day_n's average recent volatility to determine meaningful move
VOL_LOOKBACK_MULTIPLE = 10  # multiplier for number of day we want to get volatiltiy data over. VOL_LOOKBACK_MULTIPLE * timeframe


def get_target_values(timeframe: int, sigma_multiple: float = SIGMA_MULTIPLE) -> pd.DataFrame:
    """
    Get the target values (1 = significant move up, -1 = significant move down, 0 = no significant move) for the given stock data and timeframe.
    """
    print(f"Getting target values for {timeframe}-day timeframe and sigma multiple {sigma_multiple}")
    stock_data: dict[str, pd.DataFrame] = get_stock_data()
    equity_ids: dict[str, int] = get_equity_id()
    # Each ETF's target labels are stored in a separate DataFrame. Each stored in outs. Concatenated at the end.
    outs: list[pd.DataFrame] = []
    for stock, data in stock_data.items():
        equity_id: int = equity_ids[stock]
        warmup_period: int = timeframe * VOL_LOOKBACK_MULTIPLE

        # daily close-to-close moves
        daily_moves: pd.Series = data["close"].diff()  # move from day n to day n+1. diff defaults to shift(+1)

        # rolling single-day std of those moves over the lookback window * sqrt(timeframe) to adjust for move over timeframe
        rolling_vol: pd.Series = daily_moves.rolling(window=warmup_period).std() * np.sqrt(timeframe)
        significant_move: pd.Series = (rolling_vol * sigma_multiple).dropna()

        close = data["close"].reindex(significant_move.index)
        high = data["high"].reindex(significant_move.index)
        low = data["low"].reindex(significant_move.index)

        # NEED TO DERIVE LABELS FOR WHEN significant_move <= CURRENT_DATE + TIMEFRAME PRICE MOVE!!
        labels = pd.DataFrame({"equity_id": equity_id, "trade_date": close.index, "label": 0})

        # Find if upper or lower barrier is hit first, or neither
        for i in range(len(close) - timeframe):
            entry = close.iloc[i]
            upper = entry + significant_move.iloc[i]
            lower = entry - significant_move.iloc[i]
            for j in range(1, timeframe + 1):
                if i + j >= len(high):
                    break
                if high.iloc[i + j] >= upper:
                    labels.loc[labels.index[i], "label"] = 1
                    break
                if low.iloc[i + j] <= lower:
                    labels.loc[labels.index[i], "label"] = -1
                    break
        outs.append(labels)

    return pd.concat(outs, ignore_index=True)


if __name__ == "__main__":
    for timeframe in [3, 10, 20]:
        for sigma_multiple in [1.25, 1.5, 1.75]:
            target_values = get_target_values(timeframe, sigma_multiple)
            print(f"Total number of target values: {len(target_values)}")
            print(f"Total 0 classifications: {target_values[target_values['label'] == 0].shape[0]}")
            print(f"Total 1 classifications: {target_values[target_values['label'] == 1].shape[0]}")
            print(f"Total -1 classifications: {target_values[target_values['label'] == -1].shape[0]}")
            print("=" * 60)
