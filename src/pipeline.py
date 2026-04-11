import pandas as pd
import numpy as np
import joblib
import logging
import json
from datetime import datetime
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocessing import (
    optimize_memory, drop_useless_features,handle_infinite_and_nan, MissingValueHandler, 
    SkewedFeatureTransformer, CategoricalLevelManager, FrequencyEncoder
)
from feature_engineering import FeatureEngineeringTransformer, FeaturePruner


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudMLOpsPipeline:
    def __init__(self, target_col="isFraud", time_col="TransactionDT", output_dir="artifacts"):
        self.target_col = target_col
        self.time_col = time_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).resolve().parent.parent
        self.pipeline = self._build_pipeline()
        
    def _build_pipeline(self):
        return Pipeline([
            ('memory_opt', FunctionTransformer(optimize_memory)),
            ('drop_useless', FunctionTransformer(drop_useless_features)),
            ('missing_handler', MissingValueHandler(target_col=self.target_col)),
            ('skew_trans', SkewedFeatureTransformer(method='yeo-johnson')),
            ('cat_manager', CategoricalLevelManager()),
            ('feature_eng', FeatureEngineeringTransformer()),
            ('final_fillna', FunctionTransformer(handle_infinite_and_nan)),
            ('freq_encoder', FrequencyEncoder()),
            ('pruner', FeaturePruner(target_col=self.target_col, corr_threshold=0.95))
        ])

    def split_data(self, df, train_ratio=0.8):
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()
        
        logger.warning(f"Data Split: Train ({df_train[self.time_col].min()} -> {df_train[self.time_col].max()})")
        logger.warning(f"Data Split: Val   ({df_val[self.time_col].min()} -> {df_val[self.time_col].max()})")
        return df_train, df_val

    def run_train_flow(self, df_raw):
        df_train, df_val = self.split_data(df_raw)
        
        logger.warning("Fitting pipeline on Train set...")
        self.pipeline.fit(df_train)
        
        X_train_clean = self.pipeline.transform(df_train)
        X_val_clean = self.pipeline.transform(df_val)
        
        y_train = X_train_clean[self.target_col]
        y_val = X_val_clean[self.target_col]
        
        X_train_final = X_train_clean.drop(columns=[self.target_col])
        X_val_final = X_val_clean.drop(columns=[self.target_col])

        self.save_everything(X_train_final, y_train, X_val_final, y_val)
        
        return X_train_final, y_train, X_val_final, y_val
    
    def save_everything(self, X_train, y_train, X_val, y_val):
        data_dir = self.project_root / "data" / "processed"
        artifact_dir = self.project_root / "artifacts"

        data_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = artifact_dir / "fraud_pipeline.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        
        x_train_path = data_dir / "X_train.parquet"
        X_train.to_parquet(x_train_path, index=False)
        y_train.to_frame().to_parquet(data_dir / "y_train.parquet", index=False)
        X_val.to_parquet(data_dir / "X_val.parquet", index=False)
        y_val.to_frame().to_parquet(data_dir / "y_val.parquet", index=False)

        logger.warning(f"Pipeline: {pipeline_path.absolute()}")
        logger.warning(f"Data folder: {data_dir.absolute()}")

        metadata = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_count": X_train.shape[1],
            "features": X_train.columns.tolist()
        }
        with open(artifact_dir / "features_info.json", "w") as f:
            json.dump(metadata, f, indent=4)

# ==========================================
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    data_path = current_file.parent.parent / "data" / "merged_train_data.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        ops_pipe = FraudMLOpsPipeline(target_col="isFraud", time_col="TransactionDT")
        X_train, y_train, X_val, y_val = ops_pipe.run_train_flow(df)