from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering import FeatureEngineeringTransformer
from preprocessing import MissingValueHandler, drop_useless_features, optimize_memory


def test_missing_value_handler_requires_target_column(missing_value_df: pd.DataFrame) -> None:
    handler = MissingValueHandler(target_col="isFraud")

    with pytest.raises(ValueError, match="Target column isFraud"):
        handler.fit(missing_value_df.drop(columns=["isFraud"]))


def test_feature_engineering_requires_transaction_columns(feature_engineering_df: pd.DataFrame) -> None:
    transformer = FeatureEngineeringTransformer()

    with pytest.raises(KeyError):
        transformer.fit(feature_engineering_df.drop(columns=["TransactionAmt"]))

    with pytest.raises(KeyError):
        transformer.fit(feature_engineering_df.drop(columns=["card1"]))


def test_feature_engineering_rejects_non_numeric_transaction_amount(feature_engineering_df: pd.DataFrame) -> None:
    bad_df = feature_engineering_df.copy()
    bad_df["TransactionAmt"] = ["bad"] * len(bad_df)

    transformer = FeatureEngineeringTransformer()

    with pytest.raises(TypeError):
        transformer.fit(bad_df)


def test_empty_dataframe_helpers_return_empty() -> None:
    empty_df = pd.DataFrame()

    optimized = optimize_memory(empty_df.copy())
    dropped = drop_useless_features(empty_df.copy())

    assert optimized.empty
    assert dropped.empty
    assert list(optimized.columns) == []
    assert list(dropped.columns) == []