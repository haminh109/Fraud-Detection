from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sys

import __main__
from src import preprocess as preprocess_module
from src.feature_runtime import FraudFeatureBuilder

# Register old module paths for backward compatibility with pickled artifacts
sys.modules['feature_runtime'] = sys.modules['src.feature_runtime']

# Make classes available in __main__ for pickle unpickling
if hasattr(preprocess_module, "FullPreprocessor"):
    __main__.FullPreprocessor = preprocess_module.FullPreprocessor

__main__.FraudFeatureBuilder = FraudFeatureBuilder


class RawInferencePipeline:
    def __init__(
        self,
        preprocessor_path: str = "models/preprocessor_v1.pkl",
        feature_artifact_path: str = "artifacts/fe_artifact.pkl",
        model_artifact_path: str = "models/model.pkl",
    ):
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_builder = FraudFeatureBuilder.load(feature_artifact_path)
        self.model_artifact = joblib.load(model_artifact_path)

        self.model = self.model_artifact["model"]
        self.threshold = float(self.model_artifact["threshold"])
        self.model_name = self.model_artifact["model_name"]
        self.feature_names = self.model_artifact.get("feature_names")
        self.feature_name_mapping = self.model_artifact.get("feature_name_mapping", {})

    def _apply_preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust this wrapper to match your saved preprocessor object.
        """
        if hasattr(self.preprocessor, "transform_df"):
            return self.preprocessor.transform_df(raw_df)

        if hasattr(self.preprocessor, "transform"):
            transformed = self.preprocessor.transform(raw_df)
            if isinstance(transformed, pd.DataFrame):
                return transformed
            if hasattr(self.preprocessor, "get_feature_names_out"):
                cols = list(self.preprocessor.get_feature_names_out())
                return pd.DataFrame(transformed, columns=cols, index=raw_df.index)
            raise ValueError(
                "preprocessor.transform returned a non-DataFrame and no feature names are available. "
                "Refactor preprocess.py to expose a DataFrame transform."
            )

        raise ValueError("Unsupported preprocessor artifact. Expected transform_df or transform method.")

    @staticmethod
    def _sanitize_feature_name(name: str) -> str:
        import re

        name = str(name)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        renamed = {}
        for c in X.columns:
            if c in self.feature_name_mapping:
                renamed[c] = self.feature_name_mapping[c]
            else:
                renamed[c] = self._sanitize_feature_name(c)
        X = X.rename(columns=renamed)

        if self.feature_names:
            for c in self.feature_names:
                if c not in X.columns:
                    X[c] = 0.0
            extra = [c for c in X.columns if c not in self.feature_names]
            if extra:
                X = X.drop(columns=extra)
            X = X[self.feature_names]

        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        return X

    def predict_raw(self, records: list[dict[str, Any]], context: list[dict[str, Any]] | None = None):
        raw_df = pd.DataFrame(records)
        context = context or [{} for _ in range(len(raw_df))]

        preprocessed = self._apply_preprocess(raw_df)

        featured_rows = []
        for i in range(len(preprocessed)):
            one_row = preprocessed.iloc[[i]].copy()
            one_ctx = context[i] if i < len(context) else {}
            fe_row = self.feature_builder.transform(one_row, context=one_ctx)
            featured_rows.append(fe_row)

        featured = pd.concat(featured_rows, axis=0).reset_index(drop=True)
        X = self._align_features(featured)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.decision_function(X)
            proba = 1 / (1 + np.exp(-scores))

        pred = (proba >= self.threshold).astype(int)

        results = []
        for i in range(len(X)):
            results.append(
                {
                    "index": i,
                    "fraud_probability": float(proba[i]),
                    "prediction": int(pred[i]),
                    "threshold": self.threshold,
                    "model_name": self.model_name,
                }
            )
        return results