from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def missing_value_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "isFraud": [1, 1, 0, 0, 1, 0, 1, 0],
            "num_missing": [np.nan, np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0],
            "cat_missing": [None, "x", None, "y", None, "y", "x", None],
            "stable_num": [10, 11, 12, 13, 14, 15, 16, 17],
        }
    )


@pytest.fixture()
def feature_engineering_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [100, 500, 1000, 2000, 2500, 4000],
            "TransactionAmt": [10.0, 20.5, 30.0, 100.0, 110.5, 120.0],
            "card1": [1111, 1111, 1111, 2222, 2222, 2222],
            "D15": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "dist1": [0.5, 0.7, 0.9, 1.5, 1.7, 1.9],
            "V1": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "V2": [1.10, 1.20, 1.30, 1.40, 1.50, 1.60],
            "isFraud": [0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture()
def synthetic_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionDT": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
            "TransactionAmt": [10.5, 20.0, 15.0, 100.0, 110.0, 120.0, 14.0, 17.5, 130.0, 140.0, 5.0, 6.0],
            "card1": [1111, 1111, 1111, 2222, 2222, 2222, 1111, 1111, 2222, 2222, 3333, 3333],
            "D15": [1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            "dist1": [0.5, 1.0, np.nan, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, np.nan, 5.0, 5.5],
            "C1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "C2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "addr1": [100, 100, 101, 101, 102, 102, 100, 100, 101, 101, 103, 103],
            "V1": np.linspace(0.1, 1.2, 12),
            "V2": np.linspace(1.1, 2.2, 12),
            "ProductCD": ["W", "W", "W", "C", "C", "C", "W", "W", "W", "Z", "Z", None],
            "emaildomain": [
                "gmail.com",
                "gmail.com",
                "yahoo.com",
                "gmail.com",
                "yahoo.com",
                "hotmail.com",
                "gmail.com",
                "hotmail.com",
                "gmail.com",
                "proton.me",
                "proton.me",
                None,
            ],
            "useless_all_nan": [np.nan] * 12,
            "isFraud": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    )