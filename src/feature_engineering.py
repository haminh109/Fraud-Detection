import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import mutual_info_classif
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stats_ = {}
        self.q95_amt_ = None
        self.pca = None
        self.v_cols_ = None

    def fit(self, X, y=None):
        X_copy = X.copy()

        self.q95_amt_ = X_copy['TransactionAmt'].quantile(0.95)

        card1_group = X_copy.groupby('card1')
        self.stats_['amt_mean'] = card1_group['TransactionAmt'].mean()
        self.stats_['amt_std'] = card1_group['TransactionAmt'].std()
        self.stats_['amt_median'] = card1_group['TransactionAmt'].median()
        self.stats_['card1_counts'] = card1_group['TransactionAmt'].count()

        if 'D15' in X_copy.columns:
            self.stats_['d15_mean'] = card1_group['D15'].mean()
        if 'dist1' in X_copy.columns:
            self.stats_['dist1_mean'] = card1_group['dist1'].mean()

        # 3. PCA + TruncatedSVD 
        self.v_cols_ = [c for c in X_copy.columns if c.startswith('V')]
        if self.v_cols_:
            v_data = X_copy[self.v_cols_].fillna(-1).astype('float32')
            if len(v_data) > 100_000:
                sample_idx = np.random.choice(len(v_data), 80_000, replace=False)
                v_sample = v_data.iloc[sample_idx]
            else:
                v_sample = v_data

            self.pca = TruncatedSVD(n_components=2, random_state=42)
            self.pca.fit(v_sample)

        logger.info("FeatureEngineeringTransformer fitted successfully.")
        return self

    def transform(self, X):
        df = X.copy()
        new_features = {}

        new_features['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'].clip(lower=0))
        new_features['Amt_decimal'] = df['TransactionAmt'] % 1
        new_features['IsLargeTransaction'] = (df['TransactionAmt'] > self.q95_amt_).astype('int8')
        new_features['IsSmallTestTransaction'] = (df['TransactionAmt'] < 5).astype('int8')

        new_features['TransactionHour'] = (df['TransactionDT'] / 3600) % 24
        new_features['TransactionDayOfWeek'] = (df['TransactionDT'] / 86400) % 7
        new_features['IsNightTransaction'] = ((new_features['TransactionHour'] >= 0) & 
                                              (new_features['TransactionHour'] <= 5)).astype('int8')
        new_features['IsWeekendTransaction'] = (new_features['TransactionDayOfWeek'] >= 5).astype('int8')
        
        # Card1 aggregation features
        m = df['card1'].map(self.stats_['amt_mean'])
        s = df['card1'].map(self.stats_['amt_std'])
        med = df['card1'].map(self.stats_['amt_median'])

        new_features['card1_Amt_mean'] = m
        new_features['card1_Amt_std'] = s.fillna(0)
        new_features['card1_Amt_median'] = med

        new_features['AmountDeviationUser'] = df['TransactionAmt'] / (m + 1e-3)
        new_features['AmountStdScore'] = (df['TransactionAmt'] - m) / (s + 1e-3)
        new_features['AmountToMedianRatio'] = df['TransactionAmt'] / (med + 1e-3)
        new_features['CardTransactionCount'] = df['card1'].map(self.stats_['card1_counts']).fillna(1)

        # D15 & dist1 ratio
        if 'd15_mean' in self.stats_:
            new_features['D15_to_Mean_card1'] = df.get('D15', 0) / (df['card1'].map(self.stats_['d15_mean']) + 1e-3)
        if 'dist1_mean' in self.stats_:
            new_features['DistanceDeviation'] = df.get('dist1', 0) - df['card1'].map(self.stats_['dist1_mean'])

        # Velocity
        logger.warning("Calculating Transaction Velocity ...")
        
        orig_indices = np.arange(len(df))
        temp_df = pd.DataFrame({
            'card1': df['card1'].values,
            'TransactionDT': df['TransactionDT'].values,
            'orig_idx': orig_indices
        }).sort_values(['card1', 'TransactionDT'])

        times = temp_df['TransactionDT'].values
        cards = temp_df['card1'].values
        card_starts = np.searchsorted(cards, cards, side='left')

        for window, name in [(3600, 'TransactionVelocity1h'), (86400, 'TransactionVelocity24h')]:
            time_starts = np.searchsorted(times, times - window, side='left')
            final_starts = np.maximum(time_starts, card_starts)
            vel = (np.arange(len(times)) - final_starts + 1).astype('int32')
            new_features[name] = vel[np.argsort(temp_df['orig_idx'].values)]

        new_features['TimeSinceLastTransaction'] = df.groupby('card1')['TransactionDT'].diff().fillna(999999)

        if self.pca is not None and self.v_cols_:
            v_data = df[self.v_cols_].fillna(-1).astype('float32')
            v_pca = self.pca.transform(v_data)
            new_features['V_PCA_1'] = v_pca[:, 0]
            new_features['V_PCA_2'] = v_pca[:, 1]

        new_df = pd.DataFrame(new_features, index=df.index, dtype='float32')
        df = pd.concat([df, new_df], axis=1)
        
        logger.warning(f"Feature Engineering complete. Total columns: {df.shape[1]}")
        return df


class FeaturePruner(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="isFraud", corr_threshold=0.90):
        self.target_col = target_col
        self.corr_threshold = corr_threshold
        self.prune_to_drop_ = []

    def fit(self, df, y=None):
        if self.target_col not in df.columns:
            logger.warning(f"Target column {self.target_col} not found.")
            return self

        sample_size = min(80_000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)

        numeric_cols = [c for c in sample_df.select_dtypes(include=[np.number]).columns 
                       if c != self.target_col]

        protected = {
            'TransactionAmt', 'TransactionDT', 'card1', 'addr1', 'D1', 'D15',
            'TransactionVelocity1h', 'TransactionVelocity24h', 'V_PCA_1', 'V_PCA_2'
        }

        corr_matrix = sample_df[numeric_cols].corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop_corr = [col for col in upper.columns 
                       if any(upper[col] > self.corr_threshold)]

        self.prune_to_drop_ = [c for c in to_drop_corr if c not in protected]

        logger.info(f"Pruner identified {len(self.prune_to_drop_)} features to drop "
                   f"(from sample of {sample_size:,} rows).")
        return self

    def transform(self, df):
        dropped = df.drop(columns=self.prune_to_drop_, errors='ignore')
        logger.info(f"FeaturePruner dropped {len(self.prune_to_drop_)} features. "
                   f"Remaining columns: {dropped.shape[1]}")
        return dropped

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


STRONG_DROP_CANDIDATES = [
    "id_24", "id_25", "id_07", "id_08", "id_21",
    "id_26", "id_22", "id_23", "id_27"
]

TOP_MISSING_SIGNAL_COLS = [
    "R_emaildomain", "id_02", "id_15", "DeviceType"
]

TOP_V_SIGNAL_COLS = [
    "V257", "V258", "V246", "V243", "V265", "V264", "V219"
]

IMPORTANT_CATEGORICALS = [
    "ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
]


# =========================
# 1. BASIC FEATURE BUILDERS
# =========================
def add_missing_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TOP_MISSING_SIGNAL_COLS:
        if col in df.columns:
            df[f"{col}_missing_flag"] = df[col].isna().astype("int8")

    df["null_counts"] = df.isnull().sum(axis=1).astype("int16")
    df["null_count_bin"] = pd.cut(
        df["null_counts"],
        bins=[-1, 5, 15, 30, 60, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype("int8")

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionAmt" not in df.columns:
        return df

    amt = df["TransactionAmt"].clip(lower=0)

    df["TransactionAmt_Log"] = np.log1p(amt)
    df["Amt_decimal"] = df["TransactionAmt"] % 1

    if "card1" in df.columns:
        card1_group = df.groupby("card1")["TransactionAmt"]
        df["card1_Amt_mean"] = card1_group.transform("mean")
        df["card1_Amt_std"] = card1_group.transform("std")
        df["card1_Amt_median"] = card1_group.transform("median")

        df["AmountDeviationUser"] = df["TransactionAmt"] / (df["card1_Amt_mean"] + 0.001)
        df["AmountStdScore"] = (
            (df["TransactionAmt"] - df["card1_Amt_mean"]) / (df["card1_Amt_std"] + 0.001)
        )
        df["AmountToMedianRatio"] = df["TransactionAmt"] / (df["card1_Amt_median"] + 0.001)

    amt_95 = df["TransactionAmt"].quantile(0.95)
    amt_99 = df["TransactionAmt"].quantile(0.99)

    df["IsLargeTransaction"] = (df["TransactionAmt"] > amt_95).astype("int8")
    df["IsExtremeAmount"] = (df["TransactionAmt"] > amt_99).astype("int8")
    df["IsSmallTestTransaction"] = (df["TransactionAmt"] < 5).astype("int8")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TransactionDT" not in df.columns:
        return df

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    df["TransactionHour"] = ((df["TransactionDT"] / 3600) % 24).astype("float32")
    df["TransactionDayOfWeek"] = ((df["TransactionDT"] / 86400) % 7).astype("float32")

    df["IsNightTransaction"] = (
        (df["TransactionHour"] >= 0) & (df["TransactionHour"] <= 5)
    ).astype("int8")
    df["IsWeekendTransaction"] = (df["TransactionDayOfWeek"] >= 5).astype("int8")

    df["TransactionHour_sin"] = np.sin(2 * np.pi * df["TransactionHour"] / 24)
    df["TransactionHour_cos"] = np.cos(2 * np.pi * df["TransactionHour"] / 24)
    df["TransactionDay_sin"] = np.sin(2 * np.pi * df["TransactionDayOfWeek"] / 7)
    df["TransactionDay_cos"] = np.cos(2 * np.pi * df["TransactionDayOfWeek"] / 7)

    if "card1" in df.columns:
        # NaN đầu tiên có nghĩa: first transaction
        df["TimeSinceLastTransaction"] = df.groupby("card1")["TransactionDT"].diff()

        df["temp_ts"] = pd.to_datetime(df["TransactionDT"], unit="s")
        temp_df = df[["card1", "temp_ts", "TransactionDT"]].copy().set_index("temp_ts")

        df["TransactionVelocity1h"] = (
            temp_df.groupby("card1")["TransactionDT"]
            .rolling("3600s")
            .count()
            .reset_index(level=0, drop=True)
            .values
        )

        df["TransactionVelocity24h"] = (
            temp_df.groupby("card1")["TransactionDT"]
            .rolling("86400s")
            .count()
            .reset_index(level=0, drop=True)
            .values
        )

        df.drop(columns=["temp_ts"], inplace=True)

    return df


def add_card_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"card1", "TransactionAmt"}.issubset(df.columns):
        df["CardTransactionCount"] = df.groupby("card1")["TransactionAmt"].transform("count")
        df["CardissuerFrequency"] = df["card1"].map(df["card1"].value_counts(dropna=False))

    if "D1" in df.columns:
        df["DaysSinceRegistration"] = df["D1"]
        df["AccountAgeRisk"] = 1 / (df["D1"] + 1)

    if "D2" in df.columns:
        df["TimeSinceLastPurchase"] = df["D2"]

    if {"TransactionDT", "D1"}.issubset(df.columns):
        df["RegistrationToTransactionGap"] = df["TransactionDT"] - df["D1"]

    if {"D15", "card1"}.issubset(df.columns):
        df["D15_to_Mean_card1"] = df["D15"] / (
            df.groupby("card1")["D15"].transform("mean") + 0.001
        )

    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"addr1", "addr2"}.issubset(df.columns):
        df["AddrMismatch"] = (df["addr1"] != df["addr2"]).astype("int8")

    if {"addr1", "TransactionAmt"}.issubset(df.columns):
        df["AddressTransactionCount"] = df.groupby("addr1")["TransactionAmt"].transform("count")

    if {"card1", "addr1", "TransactionAmt"}.issubset(df.columns):
        df["CardAddressCombination"] = df.groupby(["card1", "addr1"])["TransactionAmt"].transform("count")
        df["card1_addr1_unique"] = df.groupby("card1")["addr1"].transform("nunique")

    if {"dist1", "card1"}.issubset(df.columns):
        df["DistanceDeviation"] = df["dist1"] - df.groupby("card1")["dist1"].transform("mean")

    if "dist1" in df.columns:
        df["IsLongDistance"] = (df["dist1"] > 100).astype("int8")
        dist_min = df["dist1"].min()
        dist_max = df["dist1"].max()
        df["DistanceRiskScore"] = (df["dist1"] - dist_min) / (dist_max - dist_min + 1e-8)

    return df


def add_email_device_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"P_emaildomain", "R_emaildomain"}.issubset(df.columns):
        df["EmailDomainMatch"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("int8")

    if "P_emaildomain" in df.columns:
        df["IsAnonymousEmail"] = df["P_emaildomain"].isin(
            ["protonmail.com", "mail.com"]
        ).astype("int8")
        df["EmailDomainFrequency"] = df["P_emaildomain"].map(
            df["P_emaildomain"].value_counts(dropna=False)
        )

    if "DeviceType" in df.columns:
        df["IsMobileDevice"] = (df["DeviceType"] == "mobile").astype("int8")

    if "DeviceInfo" in df.columns:
        df["Device_Freq"] = df["DeviceInfo"].map(df["DeviceInfo"].value_counts(dropna=False))

    if "id_31" in df.columns:
        df["id_31_device"] = df["id_31"].astype(str).str.split(" ", expand=True)[0]

    return df


def add_association_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "C5" in df.columns:
        df["CardIPCount"] = df["C5"]

    if "C7" in df.columns:
        df["AddressDeviceCount"] = df["C7"]

    if {"C1", "C2", "C3"}.issubset(df.columns):
        df["AssociationRatio"] = (df["C1"] + df["C2"]) / (df["C3"] + 0.01)
        df["TotalAssociations"] = df["C1"] + df["C2"] + df["C3"]

    if "C1" in df.columns:
        df["C1_count"] = df["C1"].map(df["C1"].value_counts(dropna=False))

    return df


def add_top_v_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TOP_V_SIGNAL_COLS:
        if col in df.columns:
            df[f"{col}_missing_flag"] = df[col].isna().astype("int8")

            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)

            df[f"{col}_high_flag"] = (df[col] > q99).astype("int8")
            df[f"{col}_low_flag"] = (df[col] < q01).astype("int8")

    return df


# =========================
# 2. TRAIN-FIT / VAL-TRANSFORM HELPERS
# =========================
def build_train_frequency_maps(train_df: pd.DataFrame) -> Dict[str, pd.Series]:
    freq_maps = {}
    for col in IMPORTANT_CATEGORICALS:
        if col in train_df.columns:
            freq_maps[col] = train_df[col].value_counts(dropna=False)
    return freq_maps


def apply_train_frequency_maps(df: pd.DataFrame, freq_maps: Dict[str, pd.Series]) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in freq_maps.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(mapping)
            df[f"{col}_freq_unseen"] = df[f"{col}_freq"].isna().astype("int8")
    return df


def fit_pca_on_train(train_df: pd.DataFrame, pca_components: int = 2) -> Tuple[PCA | None, Dict[str, float], list]:
    v_cols = [c for c in train_df.columns if c.startswith("V")]
    if len(v_cols) == 0:
        return None, {}, []

    v_train = train_df[v_cols].copy()
    v_fill_values = v_train.median().to_dict()
    v_train = v_train.fillna(v_fill_values)

    pca = PCA(n_components=pca_components, random_state=42)
    pca.fit(v_train)

    return pca, v_fill_values, v_cols


def apply_pca_transform(
    df: pd.DataFrame,
    pca: PCA | None,
    v_fill_values: Dict[str, float],
    v_cols: list,
    pca_components: int = 2
) -> pd.DataFrame:
    df = df.copy()

    if pca is None or len(v_cols) == 0:
        return df

    available_v_cols = [c for c in v_cols if c in df.columns]
    if len(available_v_cols) != len(v_cols):
        missing_cols = set(v_cols) - set(available_v_cols)
        for col in missing_cols:
            df[col] = np.nan
        available_v_cols = v_cols

    v_matrix = df[available_v_cols].copy().fillna(v_fill_values)
    v_pca = pca.transform(v_matrix)

    for i in range(pca_components):
        df[f"V_PCA_{i+1}"] = v_pca[:, i]

    return df


# =========================
# 3. MISSING-SEMANTIC PRESERVATION
# =========================
def preserve_missing_semantics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    semantic_fill_map = {
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
        # card/email/device frequency-like features
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

    for col, fill_value in semantic_fill_map.items():
        if col in df.columns:
            df[f"{col}_missing_flag"] = df[col].isna().astype("int8")
            df[col] = df[col].fillna(fill_value)

    # unseen frequency maps from train
    for col in [f"{c}_freq" for c in IMPORTANT_CATEGORICALS]:
        if col in df.columns:
            if f"{col}_unseen" not in df.columns:
                df[f"{col}_unseen"] = df[col].isna().astype("int8")
            df[col] = df[col].fillna(0.0)

    # cleanup numeric infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # any remaining numeric NaN -> sentinel with flag
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            flag_col = f"{col}_missing_flag"
            if flag_col not in df.columns:
                df[flag_col] = df[col].isna().astype("int8")
            df[col] = df[col].fillna(-999.0)

    return df


# =========================
# 4. MAIN FEATURE PIPELINE
# =========================
def base_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    drop_cols = [c for c in STRONG_DROP_CANDIDATES if c in df.columns]
    if drop_cols:
        logging.info("Dropping %d strong drop candidates", len(drop_cols))
        df.drop(columns=drop_cols, inplace=True)

    df = add_missing_signal_features(df)
    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_card_features(df)
    df = add_location_features(df)
    df = add_email_device_features(df)
    df = add_association_features(df)
    df = add_top_v_features(df)

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic-safe feature engineering (B-route).")
    parser.add_argument("--train_input_path", type=str, required=True)
    parser.add_argument("--val_input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="isFraud")
    parser.add_argument("--pca_components", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    train_path = Path(args.train_input_path)
    val_path = Path(args.val_input_path)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading train and validation data")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    target = args.target_col

    if target not in train_df.columns or target not in val_df.columns:
        raise ValueError(f"Target column '{target}' must exist in both train and validation datasets.")

    # keep target separate during FE fitting
    y_train = train_df[target].copy()
    y_val = val_df[target].copy()

    train_base = train_df.drop(columns=[target]).copy()
    val_base = val_df.drop(columns=[target]).copy()

    # ---------------- FIT ON TRAIN ----------------
    logging.info("Building base features on train")
    train_fe = base_feature_engineering(train_base)

    logging.info("Fitting train-based frequency maps")
    freq_maps = build_train_frequency_maps(train_fe)
    train_fe = apply_train_frequency_maps(train_fe, freq_maps)

    logging.info("Fitting PCA on train V-block")
    pca, v_fill_values, v_cols = fit_pca_on_train(train_fe, args.pca_components)
    train_fe = apply_pca_transform(train_fe, pca, v_fill_values, v_cols, args.pca_components)

    logging.info("Applying semantic-safe imputation on train")
    train_fe = preserve_missing_semantics(train_fe)

    # ---------------- TRANSFORM VAL ----------------
    logging.info("Building base features on validation")
    val_fe = base_feature_engineering(val_base)

    logging.info("Applying train-based frequency maps to validation")
    val_fe = apply_train_frequency_maps(val_fe, freq_maps)

    logging.info("Applying train-fitted PCA to validation")
    val_fe = apply_pca_transform(val_fe, pca, v_fill_values, v_cols, args.pca_components)

    logging.info("Applying semantic-safe imputation on validation")
    val_fe = preserve_missing_semantics(val_fe)

    # restore target
    train_fe[target] = y_train.values
    val_fe[target] = y_val.values

    # split X/y
    X_train = train_fe.drop(columns=[target])
    y_train = train_fe[target]

    X_val = val_fe.drop(columns=[target])
    y_val = val_fe[target]

    # sanity checks
    logging.info("Train FE shape: %s", X_train.shape)
    logging.info("Val FE shape: %s", X_val.shape)
    logging.info("Remaining NaN in X_train: %d", int(X_train.isna().sum().sum()))
    logging.info("Remaining NaN in X_val: %d", int(X_val.isna().sum().sum()))

    # save
    X_train.to_parquet(output_dir / "X_train.parquet", index=False)
    y_train.to_frame().to_parquet(output_dir / "y_train.parquet", index=False)

    X_val.to_parquet(output_dir / "X_val.parquet", index=False)
    y_val.to_frame().to_parquet(output_dir / "y_val.parquet", index=False)

    logging.info("Saved FE outputs to %s", output_dir)


if __name__ == "__main__":
    main()
