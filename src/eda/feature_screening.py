import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from src.feature_engineering.generate_all_features import get_ti_features
from src.utils import get_db_conn


def calculate_mi(feature_data: pd.DataFrame, target_labels: pd.Series) -> pd.Series:
    """
    Calculates the mutual information scores for each feature.
    """
    mi_scores = mutual_info_classif(feature_data, target_labels, discrete_features="auto", random_state=42)
    mi_scores = pd.Series(mi_scores, index=feature_data.columns)
    return mi_scores


def calculate_spearman(feature_data: pd.DataFrame, target_labels: pd.Series) -> pd.Series:
    """
    Calculates the Spearman's rank correlation scores for each feature.
    """
    spearman_scores = {feature_name: spearmanr(feature_data[feature_name], target_labels)[0] for feature_name in feature_data.columns}
    return pd.Series(spearman_scores)


def intra_feature_correlation(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix for the features.
    """
    correlation_matrix = feature_data.corr(method="spearman")
    return correlation_matrix


def candidates_to_remove(mi_scores: pd.Series, spearman_scores: pd.Series, corr_matrix: pd.DataFrame) -> set[str]:
    """
    Determines the features to remove based on the MI, Spearman's rank correlation, and correlation matrix.
    """
    candidates: set[str] = set()

    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(float(corr_matrix.iloc[i, j])) >= 0.9:
                worse = cols[i] if mi_scores[cols[i]] <= mi_scores[cols[j]] else cols[j]
                candidates.add(worse)

    for feature, mi_score, spearman_score in zip(mi_scores.index, mi_scores, spearman_scores):
        if mi_score < 0.0001 and abs(spearman_score) < 0.03:
            candidates.add(feature)
    return candidates


def run_eda() -> dict[int, dict[str, set[str]]]:
    """
    Runs the EDA process and returns the final features to keep for each TI/timeframe combination.
    """
    all_features: dict[int, dict[str, pd.DataFrame]] = get_ti_features(get_db_conn())

    final_candidates: dict[int, dict[str, set[str]]] = {}

    for timeframe, ti_feature_map in all_features.items():
        final_candidates[timeframe] = {}
        for ti_name, feature_data in ti_feature_map.items():
            final_candidates[timeframe][ti_name] = set()
            print(f"============== {ti_name} ({timeframe}-day timeframe) ==============")
            target_labels: pd.Series = feature_data["label"]
            feature_data: pd.DataFrame = feature_data.drop(columns=["label", "equity_id", "trade_date"])

            mi_scores: pd.Series = calculate_mi(feature_data, target_labels)
            mi_scores.to_excel(f"src/eda/mutual_info_series/mi_scores_{ti_name}_{timeframe}.xlsx", index=True)

            spearman_scores: pd.Series = calculate_spearman(feature_data, target_labels)
            spearman_scores.to_excel(f"src/eda/spearman_score_series/spearman_scores_{ti_name}_{timeframe}.xlsx", index=True)
            correlation_matrix: pd.DataFrame = intra_feature_correlation(feature_data)

            candidates: set[str] = candidates_to_remove(mi_scores, spearman_scores, correlation_matrix)
            pd.DataFrame({"feature": list(candidates)}).to_csv(f"src/eda/candidates_to_remove/candidates_{ti_name}_{timeframe}.csv", index=True)
            final_candidates[timeframe][ti_name] = set(col for col in feature_data.columns if col not in candidates)
            print(f"Candidates to remove:\n {candidates}\n")

            print(f"MI scores:\n {mi_scores}\n")
            print(f"Spearman scores:\n {spearman_scores}\n")
            print("Correlation matrix:\n")

            sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0, vmin=-1, vmax=1)
            plt.title(f"Intra-Group Correlation ({timeframe}-day timeframe)")
            plt.tight_layout()
            plt.savefig(f"src/eda/intra_corr_heatmaps/intra_feature_correlation_{ti_name}_{timeframe}.png")
            plt.close()
            print("=" * 60)
            print("\n")
    return final_candidates


if __name__ == "__main__":
    final_candidates = run_eda()
    print(json.dumps(final_candidates, indent=4))
