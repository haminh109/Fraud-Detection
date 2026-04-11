import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_infinite_and_nan(df):
    return df.replace([np.float64('inf'), np.float64('-inf')], -999).fillna(-999)

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
            
    return df

def drop_useless_features(df: pd.DataFrame) -> pd.DataFrame:
    null_thresh = 0.99
    nan_cols = [c for c in df.columns if df[c].isna().mean() > null_thresh]
    existing_drop = [c for c in nan_cols if c in df.columns]
    df = df.drop(columns=existing_drop)
    logger.info(f"Dropped {len(existing_drop)} useless columns.")
    return df

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="isFraud", indicator_threshold=0.01,
                 abs_gap_threshold=0.02, top_k_missing=80):
        self.target_col = target_col
        self.indicator_threshold = indicator_threshold
        self.abs_gap_threshold = abs_gap_threshold
        self.top_k_missing = top_k_missing

        self.top_features_: List[str] = []
        self.impute_stats_: Dict = {}
        self.indicator_cols_: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        if self.target_col not in df.columns:
            raise ValueError(f"Target column {self.target_col} must be in DataFrame during FIT.")

        df_copy = df.copy()
        fraud_mask = df_copy[self.target_col] == 1
        missing = df_copy.isnull()
        rows = []

        for col in df_copy.columns:
            if col == self.target_col or df_copy[col].isna().mean() < 0.001:
                continue

            miss_rate = missing[col].mean()
            miss_fraud = missing[fraud_mask][col].mean() if fraud_mask.any() else 0.0
            miss_non = missing[~fraud_mask][col].mean() if (~fraud_mask).any() else 0.0
            abs_gap = abs(miss_fraud - miss_non)

            is_missing = missing[col]
            fraud_rate_miss = df_copy.loc[is_missing, self.target_col].mean() if is_missing.any() else 0.0
            fraud_rate_present = df_copy.loc[~is_missing, self.target_col].mean()
            fraud_rate_gap = abs(fraud_rate_miss - fraud_rate_present)

            corr_val = 0.0
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                corr_val = df_copy[col].corr(df_copy[self.target_col], method='spearman')
                corr_val = abs(corr_val) if pd.notna(corr_val) else 0.0

            score = (0.35 * abs_gap + 0.30 * fraud_rate_gap + 0.25 * corr_val + 0.10 * miss_rate)
            rows.append({"feature": col, "score": score, "miss_rate": miss_rate, "abs_gap": abs_gap})

        ranking = pd.DataFrame(rows).sort_values("score", ascending=False)
        self.top_features_ = ranking.head(self.top_k_missing)["feature"].tolist()

        for col in self.top_features_:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                self.impute_stats_[col] = {
                    "overall": df_copy[col].median() if pd.notna(df_copy[col].median()) else -999.0
                }
            else:
                mode_res = df_copy[col].mode()
                self.impute_stats_[col] = mode_res[0] if not mode_res.empty else "MISSING"

        self.indicator_cols_ = [
            col for col in self.top_features_
            if df_copy[col].isna().mean() >= self.indicator_threshold or
               abs(df_copy.loc[fraud_mask, col].isna().mean() - df_copy.loc[~fraud_mask, col].isna().mean()) >= self.abs_gap_threshold
        ]
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for col in self.indicator_cols_:
            if col in df.columns:
                df[f"{col}_isna"] = df[col].isna().astype("int8")

        for col, stats in self.impute_stats_.items():
            if col in df.columns:
                if isinstance(stats, dict):  # Numeric
                    df[col] = df[col].fillna(stats["overall"]).astype("float32")
                else:  # Categorical
                    df[col] = df[col].fillna(stats).astype("category")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(-999).astype("float32")
        return df

class SemanticSentinelFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.semantic_fill_map = {
            # time/history
            "TimeSinceLastTransaction": 999999.0,
            # group stats
            "card1_Amt_std": 0.0,
            "AmountStdScore": -1.0,
            "card1_Amt_mean": -1.0,
            "card1_Amt_median": -1.0,
            "AmountDeviationUser": -1.0,
            "AmountToMedianRatio": -1.0,
            # distance/location
            "DistanceDeviation": -999.0,
            "DistanceRiskScore": -1.0,
            # frequency-like features
            "CardissuerFrequency": 0.0,
            "AddressTransactionCount": 0.0,
            "CardAddressCombination": 0.0,
            "card1_addr1_unique": 0.0,
            "EmailDomainFrequency": 0.0,
            "Device_Freq": 0.0,
            "C1_count": 0.0,
            # D features
            "D15_to_Mean_card1": -1.0,
        }

    def fit(self, X, y=None):
        # Chỉ cần lưu map, không cần fit gì thêm
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for col, fill_value in self.semantic_fill_map.items():
            if col in df.columns:
                flag_col = f"{col}_missing_flag"
                if flag_col not in df.columns:
                    df[flag_col] = df[col].isna().astype("int8")
                
                df[col] = df[col].fillna(fill_value)

        df = handle_infinite_and_nan(df)
        return df
    
    
class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='yeo-johnson', log_cols=None):
        self.method = method
        self.log_cols = log_cols or ['TransactionAmt', 'C1', 'C2']
        self.transformers_ = {}

    def fit(self, X, y=None):
        for col in self.log_cols:
            if col in X.columns:
                pt = PowerTransformer(method=self.method, standardize=True)
                pt.fit(X[[col]].fillna(X[col].median()))
                self.transformers_[col] = pt
        return self

    def transform(self, X):
        X = X.copy()
        for col, pt in self.transformers_.items():
            if col in X.columns:                
                fill_val = X[col].median()
                X[col] = pt.transform(X[[col]].fillna(fill_val))
        
        return X

class CategoricalLevelManager(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.0005):
        self.min_freq = min_freq
        self.common_labels_ = {}

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            freq = X[col].value_counts(normalize=True, dropna=True)
            self.common_labels_[col] = freq[freq >= self.min_freq].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col, common in self.common_labels_.items():
            if col in X.columns:
                series = X[col].astype(str)
                is_null = X[col].isna() | (series == 'nan') | (series == 'None')
                X[col] = np.where(is_null, "MISSING",
                                  np.where(series.isin(common), series, "OTHER"))
        return X

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=0.0001):
        self.min_freq = min_freq
        self.freq_map_ = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(exclude=[np.number]).columns:
            freq = X[col].value_counts(normalize=True, dropna=False)
            self.freq_map_[col] = freq[freq >= self.min_freq]
        return self

    def transform(self, X):
        X = X.copy()
        for col, freq_map in self.freq_map_.items():
            if col in X.columns:
                X[col] = X[col].map(freq_map).fillna(0.0).astype("float32")
        return X