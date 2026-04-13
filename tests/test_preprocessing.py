from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import FeatureEngineeringTransformer, FeaturePruner
from pipeline import FraudMLOpsPipeline
from preprocessing import (
    CategoricalLevelManager,
    FrequencyEncoder,
    MissingValueHandler,
    SkewedFeatureTransformer,
    drop_useless_features,
    handle_infinite_and_nan,
    optimize_memory,
)


def test_handle_infinite_and_nan_replaces_bad_values() -> None:
    df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, np.nan]})

    result = handle_infinite_and_nan(df)

    assert result["a"].tolist() == [1.0, -999.0, -999.0, -999.0]


def test_optimize_memory_downcasts_numeric_and_converts_objects() -> None:
    df = pd.DataFrame(
        {
            "small_int": pd.Series([1, 2, 3], dtype="int64"),
            "float_col": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "text": ["a", "b", "c"],
        }
    )

    result = optimize_memory(df)

    assert str(result["small_int"].dtype) == "int8"
    assert str(result["float_col"].dtype) == "float32"
    assert str(result["text"].dtype) == "category"


def test_drop_useless_features_removes_columns_above_missing_threshold() -> None:
    df = pd.DataFrame(
        {
            "keep": [1.0, 2.0, 3.0],
            "drop_me": [np.nan, np.nan, np.nan],
        }
    )

    result = drop_useless_features(df)

    assert "drop_me" not in result.columns
    assert "keep" in result.columns


def test_missing_value_handler_adds_indicators_and_imputes(missing_value_df: pd.DataFrame) -> None:
    handler = MissingValueHandler(target_col="isFraud", top_k_missing=10)
    handler.fit(missing_value_df)

    transformed = handler.transform(missing_value_df)

    assert "num_missing_isna" in transformed.columns
    assert "cat_missing_isna" in transformed.columns
    assert not transformed["num_missing"].isna().any()
    assert not transformed["cat_missing"].isna().any()
    assert set(transformed["num_missing_isna"].unique()) <= {0.0, 1.0}
    assert set(transformed["cat_missing_isna"].unique()) <= {0.0, 1.0}
    assert str(transformed["num_missing"].dtype) == "float32"


def test_skewed_feature_transformer_preserves_shape_and_removes_nans() -> None:
    df = pd.DataFrame(
        {
            "TransactionAmt": [1.0, 2.0, np.nan, 8.0, 16.0],
            "C1": [1.0, 1.0, 2.0, np.nan, 3.0],
            "C2": [0.0, 5.0, 10.0, 15.0, np.nan],
            "other": [100, 200, 300, 400, 500],
        }
    )

    transformer = SkewedFeatureTransformer()
    transformer.fit(df)
    transformed = transformer.transform(df)

    assert transformed.shape == df.shape
    assert not transformed[["TransactionAmt", "C1", "C2"]].isna().any().any()
    assert not np.allclose(
        transformed["TransactionAmt"].to_numpy(),
        df["TransactionAmt"].fillna(df["TransactionAmt"].median()).to_numpy(),
    )


def test_categorical_level_manager_groups_rare_and_unseen_labels() -> None:
    train = pd.DataFrame({"ProductCD": ["A", "A", "A", "B", None]})
    test = pd.DataFrame({"ProductCD": ["A", "B", "Z", None]})

    manager = CategoricalLevelManager(min_freq=0.5)
    manager.fit(train)
    transformed = manager.transform(test)

    assert transformed["ProductCD"].tolist() == ["A", "OTHER", "OTHER", "MISSING"]


def test_frequency_encoder_maps_categories_and_unseen_to_zero() -> None:
    train = pd.DataFrame({"ProductCD": ["W", "W", "C", None]})
    test = pd.DataFrame({"ProductCD": ["W", "Z", None]})

    encoder = FrequencyEncoder(min_freq=0.2)
    encoder.fit(train)
    transformed = encoder.transform(test)

    assert transformed["ProductCD"].dtype == np.float32
    assert transformed["ProductCD"].tolist() == [0.5, 0.0, 0.25]


def test_feature_engineering_adds_expected_columns(feature_engineering_df: pd.DataFrame) -> None:
    transformer = FeatureEngineeringTransformer()
    transformer.fit(feature_engineering_df)

    transformed = transformer.transform(feature_engineering_df)

    expected_columns = {
        "TransactionAmt_Log",
        "Amt_decimal",
        "IsLargeTransaction",
        "TransactionHour",
        "TransactionVelocity1h",
        "TransactionVelocity24h",
        "TimeSinceLastTransaction",
        "V_PCA_1",
        "V_PCA_2",
    }

    assert expected_columns.issubset(set(transformed.columns))
    assert transformed.shape[0] == feature_engineering_df.shape[0]
    assert not transformed[list(expected_columns)].isna().any().any()


def test_feature_pruner_drops_highly_correlated_non_protected_feature() -> None:
    df = pd.DataFrame(
        {
            "TransactionAmt": np.arange(1, 11, dtype=float),
            "duplicate_amt": np.arange(1, 11, dtype=float),
            "addr1": np.arange(101, 111, dtype=float),
            "V_PCA_1": np.linspace(0.1, 1.0, 10),
            "isFraud": [0, 1] * 5,
        }
    )

    pruner = FeaturePruner(target_col="isFraud", corr_threshold=0.95)
    pruner.fit(df)
    transformed = pruner.transform(df)

    assert "duplicate_amt" in pruner.prune_to_drop_
    assert "TransactionAmt" not in pruner.prune_to_drop_
    assert "duplicate_amt" not in transformed.columns
    assert "TransactionAmt" in transformed.columns


def test_pipeline_build_contains_expected_steps(tmp_path) -> None:
    pipeline_obj = FraudMLOpsPipeline(output_dir=tmp_path)

    assert list(pipeline_obj.pipeline.named_steps) == [
        "memory_opt",
        "drop_useless",
        "missing_handler",
        "skew_trans",
        "cat_manager",
        "feature_eng",
        "final_fillna",
        "freq_encoder",
        "pruner",
    ]


def test_pipeline_run_train_flow_returns_clean_features_and_target(
    monkeypatch,
    synthetic_raw_df: pd.DataFrame,
    tmp_path,
) -> None:
    monkeypatch.setattr(FraudMLOpsPipeline, "save_everything", lambda self, *args, **kwargs: None)

    pipeline_obj = FraudMLOpsPipeline(output_dir=tmp_path)
    X_train, y_train, X_val, y_val = pipeline_obj.run_train_flow(synthetic_raw_df)

    ordered = synthetic_raw_df.sort_values("TransactionDT").reset_index(drop=True)
    expected_y_train = ordered.iloc[:9]["isFraud"].astype("float32").reset_index(drop=True)
    expected_y_val = ordered.iloc[9:]["isFraud"].astype("float32").reset_index(drop=True)

    assert len(X_train) == 9
    assert len(y_train) == 9
    assert len(X_val) == 3
    assert len(y_val) == 3
    assert "isFraud" not in X_train.columns
    assert "isFraud" not in X_val.columns
    assert list(X_train.columns) == list(X_val.columns)
    assert "useless_all_nan" not in X_train.columns
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes)
    assert y_train.reset_index(drop=True).equals(expected_y_train)
    assert y_val.reset_index(drop=True).equals(expected_y_val)