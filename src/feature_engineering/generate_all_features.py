import pandas as pd
from mysql.connector.abstracts import MySQLConnectionAbstract
from mysql.connector.pooling import PooledMySQLConnection
from mysql.connector.types import RowItemType

from src.data_pipeline.generate_target_values import get_target_values
from src.feature_engineering.generate_adx_features import AdxFeatureGenerator
from src.feature_engineering.generate_atr_features import AtrFeatureGenerator
from src.feature_engineering.generate_bbands_features import BbandsFeatureGenerator
from src.feature_engineering.generate_ema_features import EmaFeatureGenerator
from src.feature_engineering.generate_macd_features import MacdFeatureGenerator
from src.feature_engineering.generate_obv_features import ObvFeatureGenerator
from src.feature_engineering.generate_rsi_features import RsiFeatureGenerator
from src.feature_engineering.generate_sma_features import SmaFeatureGenerator
from src.feature_engineering.generate_stoch_features import StochFeatureGenerator
from src.feature_engineering.generate_vol_roc_features import VolRocFeatureGenerator

TI_TABLE_NAMES: dict[str, str] = {
    "OHLCV": "ohlcv_daily",
    "RSI": "rsi_daily",
    "STOCH": "stoch_daily",
    "SMA": "sma_daily",
    "EMA": "ema_daily",
    "MACD": "macd_daily",
    "ADX": "adx_daily",
    "ATR": "atr_daily",
    "BBANDS": "bbands_daily",
    "OBV": "obv_daily",
    "VOL_ROC": "volume_activity_daily",
}

_OHLCV_COLS_NEEDED: dict[str, list[str]] = {
    "SMA": ["close"],
    "EMA": ["close"],
    "ATR": ["close"],
    "BBANDS": ["close"],
    "OBV": ["close", "volume"],
    "VOL_ROC": ["volume"],
}


def get_raw_ti_data(
    conn: PooledMySQLConnection | MySQLConnectionAbstract,
) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Returns {timeframe: {ti_name: DataFrame}} where each DataFrame holds
    all ETFs' raw indicator values for that TI/timeframe combination.
    OHLCV is also returned under the key "OHLCV" (timeframe-independent).

    TIs that need extra ohlcv columns (close/volume) have them merged in
    from the OHLCV data on (equity_id, trade_date).
    """
    cursor = conn.cursor(dictionary=True)
    result: dict[int, dict[str, pd.DataFrame]] = {}

    cursor.execute("SELECT * FROM ohlcv_daily")
    rows: list[dict[str, RowItemType | None]] = cursor.fetchall()  # type: ignore[type-arg]
    ohlcv = pd.DataFrame(rows, columns=list(rows[0].keys()))
    result[3] = {"OHLCV": ohlcv}
    result[10] = {"OHLCV": ohlcv}
    result[20] = {"OHLCV": ohlcv}

    for tf in (3, 10, 20):
        for ti_name, table in TI_TABLE_NAMES.items():
            if ti_name == "OHLCV":
                continue

            cursor.execute(
                "SELECT ti_def_id FROM ti_definitions WHERE ti_name = %s AND time_frame = %s",
                (ti_name, tf),
            )
            row: dict[str, RowItemType | None] | None = cursor.fetchone()  # type: ignore[type-arg]
            if row is None:
                raise ValueError(f"No ti_def_id found for {ti_name} / timeframe {tf}")

            ti_def_id: int = int(row["ti_def_id"])  # type: ignore[arg-type]
            cursor.execute(f"SELECT * FROM {table} WHERE ti_def_id = %s", (ti_def_id,))
            rows: list[dict[str, RowItemType | None]] = cursor.fetchall()  # type: ignore[type-arg]
            df = pd.DataFrame(rows, columns=list(rows[0].keys()))
            df.drop(columns=["ti_def_id"], inplace=True)

            extra_cols = _OHLCV_COLS_NEEDED.get(ti_name)
            if extra_cols:
                ohlcv_df = result[tf]["OHLCV"]
                merge_cols = ["equity_id", "trade_date"] + extra_cols
                df = df.merge(ohlcv_df[merge_cols], on=["equity_id", "trade_date"], how="left")

            result[tf][ti_name] = df

    return result


def get_ti_features(conn: PooledMySQLConnection | MySQLConnectionAbstract) -> dict[int, dict[str, pd.DataFrame]]:
    """
    Returns {timeframe: {ti_name: features_DataFrame}} where each DataFrame
    holds all ETFs' engineered features for that TI/timeframe combination and target labels.
    """
    raw_data = get_raw_ti_data(conn)
    feature_sets = {
        3: {
            "rsi": RsiFeatureGenerator(raw_data[3]["RSI"], 3).generate(),
            "sma": SmaFeatureGenerator(raw_data[3]["SMA"], 3).generate(),
            "ema": EmaFeatureGenerator(raw_data[3]["EMA"], 3).generate(),
            "macd": MacdFeatureGenerator(raw_data[3]["MACD"], 3).generate(),
            "adx": AdxFeatureGenerator(raw_data[3]["ADX"], 3).generate(),
            "stoch": StochFeatureGenerator(raw_data[3]["STOCH"], 3).generate(),
            "obv": ObvFeatureGenerator(raw_data[3]["OBV"], 3).generate(),
            "atr": AtrFeatureGenerator(raw_data[3]["ATR"], 3).generate(),
            "bbands": BbandsFeatureGenerator(raw_data[3]["BBANDS"], 3).generate(),
            "vol_roc": VolRocFeatureGenerator(raw_data[3]["VOL_ROC"], 3).generate(),
        },
        10: {
            "rsi": RsiFeatureGenerator(raw_data[10]["RSI"], 10).generate(),
            "sma": SmaFeatureGenerator(raw_data[10]["SMA"], 10).generate(),
            "ema": EmaFeatureGenerator(raw_data[10]["EMA"], 10).generate(),
            "macd": MacdFeatureGenerator(raw_data[10]["MACD"], 10).generate(),
            "adx": AdxFeatureGenerator(raw_data[10]["ADX"], 10).generate(),
            "stoch": StochFeatureGenerator(raw_data[10]["STOCH"], 10).generate(),
            "obv": ObvFeatureGenerator(raw_data[10]["OBV"], 10).generate(),
            "atr": AtrFeatureGenerator(raw_data[10]["ATR"], 10).generate(),
            "bbands": BbandsFeatureGenerator(raw_data[10]["BBANDS"], 10).generate(),
            "vol_roc": VolRocFeatureGenerator(raw_data[10]["VOL_ROC"], 10).generate(),
        },
        20: {
            "rsi": RsiFeatureGenerator(raw_data[20]["RSI"], 20).generate(),
            "sma": SmaFeatureGenerator(raw_data[20]["SMA"], 20).generate(),
            "ema": EmaFeatureGenerator(raw_data[20]["EMA"], 20).generate(),
            "macd": MacdFeatureGenerator(raw_data[20]["MACD"], 20).generate(),
            "adx": AdxFeatureGenerator(raw_data[20]["ADX"], 20).generate(),
            "stoch": StochFeatureGenerator(raw_data[20]["STOCH"], 20).generate(),
            "obv": ObvFeatureGenerator(raw_data[20]["OBV"], 20).generate(),
            "atr": AtrFeatureGenerator(raw_data[20]["ATR"], 20).generate(),
            "bbands": BbandsFeatureGenerator(raw_data[20]["BBANDS"], 20).generate(),
            "vol_roc": VolRocFeatureGenerator(raw_data[20]["VOL_ROC"], 20).generate(),
        },
    }

    feature_to_raw_key = {
        "rsi": "RSI",
        "sma": "SMA",
        "ema": "EMA",
        "macd": "MACD",
        "adx": "ADX",
        "stoch": "STOCH",
        "obv": "OBV",
        "atr": "ATR",
        "bbands": "BBANDS",
        "vol_roc": "VOL_ROC",
    }

    for timeframe, ti_feature_map in feature_sets.items():
        target = get_target_values(timeframe)
        target_labels = target["label"].reset_index(drop=True)

        for ti_name, feature_df in ti_feature_map.items():
            raw_key = feature_to_raw_key[ti_name]
            key_cols = raw_data[timeframe][raw_key][["equity_id", "trade_date"]].reset_index(drop=True)
            joined = pd.concat([key_cols, feature_df.reset_index(drop=True), target_labels], axis=1)
            ti_feature_map[ti_name] = joined.dropna().reset_index(drop=True)

    return feature_sets
