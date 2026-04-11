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