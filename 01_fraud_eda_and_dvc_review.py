# %% [markdown]
# # Fraud Detection EDA + DVC Review (for `haminh109/Fraud-Detection`)
#
# ## Objective
#
# This notebook is built for a **real-world fraud detection EDA** on the current repository structure of your project.
#
# It focuses on:
#
# 1. **Exploratory Data Analysis (EDA)** on the training data
# 2. Inspecting the dataset in a way that is useful for later fraud modeling
# 3. Evaluating whether the project should adopt **DVC (Data Version Control)**
# 4. Providing a **practical DVC rollout plan** that fits the current repo
#
# ## Modeling metrics to keep in mind
#
# For fraud detection, the metrics that matter most in the next stage are:
#
# - **Recall**: missing fraud cases is expensive
# - **Precision**: too many false alarms hurt trust and operations
# - **F1-score**: balances Recall and Precision at a selected threshold
# - **PR-AUC**: especially important in heavily imbalanced fraud data
#
# > Accuracy is **not** a useful primary metric here because fraud datasets are usually extremely imbalanced.
#
# ## Assumptions used in this notebook
#
# This notebook assumes the repository currently contains:
#
# - `raw_data/train_transaction.csv.gz` → main labeled training table
# - `raw_data/train_identity.csv` → auxiliary identity table
# - `raw_data/test_transaction.csv.gz`
# - `raw_data/test_identity.csv`
# - `raw_data/sample_submission.csv`
#
# And also assumes:
#
# - `train_transaction` contains target column **`isFraud`**
# - `TransactionID` is the join key between transaction and identity tables
# - `TransactionDT` is available as a relative time-like column
#
# If your file paths or target column names are different, edit the **Config + path resolution** cell below.

# %% [markdown]
# ## Why this notebook uses `train_transaction` as the primary EDA table
#
# For this repository, the most practical setup is:
#
# - Use **`train_transaction`** as the primary supervised dataset because it contains the target `isFraud`
# - Use **`train_identity`** as an auxiliary table to analyze identity coverage and identity-related patterns
# - Use **`test_transaction`** only for lightweight time-range comparison when useful
#
# This keeps the EDA focused on the labeled data while still surfacing data patterns that matter for later modeling.

# %%
from pathlib import Path
import math
import os
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 120)

RANDOM_STATE = 42
MAX_PLOT_ROWS = 120_000
MAX_COMPARE_PER_CLASS = 20_000
TOP_N_MISSING = 25
TOP_N_CORR = 20
TARGET_CANDIDATES = ["isFraud", "target", "label", "fraud"]

# %%
def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024


def locate_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / "raw_data").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repository root containing 'raw_data/'. "
        "Run this notebook from the repo root or from a subfolder inside the repo."
    )


def resolve_existing_path(directory: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = directory / name
        if path.exists():
            return path
    return None


def reduce_mem_usage(df: pd.DataFrame, convert_object_to_category: bool = True) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)

        elif pd.api.types.is_float_dtype(col_type):
            c_min = df[col].min(skipna=True)
            c_max = df[col].max(skipna=True)

            if pd.notna(c_min) and pd.notna(c_max):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

        elif convert_object_to_category and pd.api.types.is_object_dtype(col_type):
            nunique = df[col].nunique(dropna=False)
            if nunique / max(len(df), 1) < 0.50:
                df[col] = df[col].astype("category")

    return df


def detect_target_column(df: pd.DataFrame, candidates: list[str] | None = None) -> str | None:
    candidates = TARGET_CANDIDATES if candidates is None else candidates
    for col in candidates:
        if col in df.columns:
            return col
    return None


def make_feature_lists(df: pd.DataFrame, target_col: str | None = None):
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    return numeric_cols, categorical_cols


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = missing_count / len(df)
    non_null_count = len(df) - missing_count

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values,
            "non_null_count": non_null_count.values,
            "dtype": df.dtypes.astype(str).values,
        }
    ).sort_values(["missing_pct", "missing_count"], ascending=[False, False])
    return summary.reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    max_rows: int = 120_000,
    random_state: int = 42,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()

    class_props = df[target_col].value_counts(normalize=True, dropna=False)
    pieces = []

    for cls, frac in class_props.items():
        cls_df = df[df[target_col] == cls]
        n = max(1, int(round(frac * max_rows)))
        n = min(n, len(cls_df))
        pieces.append(cls_df.sample(n=n, random_state=random_state))

    sampled = pd.concat(pieces, axis=0)

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)

    return sampled.sample(frac=1.0, random_state=random_state)


def balanced_class_sample(
    df: pd.DataFrame,
    target_col: str,
    max_per_class: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    pieces = []
    for cls, cls_df in df.groupby(target_col):
        n = min(len(cls_df), max_per_class)
        if len(cls_df) > n:
            pieces.append(cls_df.sample(n=n, random_state=random_state))
        else:
            pieces.append(cls_df.copy())

    return pd.concat(pieces, axis=0).sample(frac=1.0, random_state=random_state)


def top_numeric_by_target_corr(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 10,
    min_non_null_ratio: float = 0.30,
    exclude: set[str] | None = None,
) -> pd.Series:
    exclude = set() if exclude is None else set(exclude)

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in exclude and col != target_col
    ]

    eligible_cols = [
        col for col in numeric_cols
        if df[col].notna().mean() >= min_non_null_ratio and df[col].nunique(dropna=True) > 2
    ]

    if not eligible_cols:
        return pd.Series(dtype=float)

    corr = df[eligible_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    return corr.head(top_n)


def choose_numeric_features(df: pd.DataFrame, target_col: str, n: int = 12) -> list[str]:
    priority = [
        "TransactionAmt",
        "TransactionDT",
        "dist1",
        "dist2",
        "C1",
        "C2",
        "C5",
        "C13",
        "D1",
        "D2",
        "D4",
        "D10",
        "D15",
    ]

    selected = [col for col in priority if col in df.columns and col != target_col]
    exclude = {"TransactionID"}

    ranked = top_numeric_by_target_corr(
        df,
        target_col=target_col,
        top_n=50,
        min_non_null_ratio=0.20,
        exclude=exclude,
    )

    for col in ranked.index.tolist():
        if col not in selected:
            selected.append(col)
        if len(selected) >= n:
            break

    return selected[:n]


def high_corr_pairs(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_corr"})
        .sort_values("abs_corr", ascending=False)
        .reset_index(drop=True)
    )
    return pairs[pairs["abs_corr"] >= threshold]


def category_profile(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    min_count: int = 200,
    top_n: int = 15,
    sort_by: str = "count",
) -> pd.DataFrame:
    temp = df[[feature, target_col]].copy()
    temp[feature] = temp[feature].astype("object").where(temp[feature].notna(), "__MISSING__")

    profile = (
        temp.groupby(feature, dropna=False)[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
        .reset_index()
    )

    profile = profile[profile["count"] >= min_count]

    if sort_by not in {"count", "fraud_rate"}:
        sort_by = "count"

    profile = profile.sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    return profile


def quantile_fraud_table(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    q: int = 10,
) -> pd.DataFrame | None:
    temp = df[[feature, target_col]].dropna().copy()
    if temp.empty or temp[feature].nunique() < 2:
        return None

    temp["bin"] = pd.qcut(temp[feature], q=min(q, temp[feature].nunique()), duplicates="drop")

    out = (
        temp.groupby("bin", observed=False)[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "fraud_rate"})
        .reset_index()
    )

    out["bin"] = out["bin"].astype(str)
    return out


def iqr_outlier_summary(df: pd.DataFrame, features: list[str], target_col: str) -> pd.DataFrame:
    records = []

    for col in features:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        if len(s) == 0 or s.nunique() < 5:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (df[col] < lower) | (df[col] > upper)
        if outlier_mask.sum() == 0:
            continue

        fraud_rate_outlier = df.loc[outlier_mask, target_col].mean()
        fraud_rate_inlier = df.loc[~outlier_mask & df[col].notna(), target_col].mean()

        records.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_share": outlier_mask.mean(),
                "fraud_rate_outlier": fraud_rate_outlier,
                "fraud_rate_inlier": fraud_rate_inlier,
                "fraud_rate_lift": (
                    fraud_rate_outlier / fraud_rate_inlier
                    if pd.notna(fraud_rate_outlier) and pd.notna(fraud_rate_inlier) and fraud_rate_inlier > 0
                    else np.nan
                ),
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("fraud_rate_lift", ascending=False).reset_index(drop=True)


def get_dvc_version() -> str | None:
    if shutil.which("dvc") is None:
        return None
    try:
        return subprocess.check_output(["dvc", "--version"], text=True).strip()
    except Exception:
        return "installed but version check failed"

def class_boxplot(ax, df: pd.DataFrame, feature_col: str, target_col: str, title: str, ylabel: str):
    groups = []
    labels = []
    for cls in sorted(df[target_col].dropna().unique().tolist()):
        series = df.loc[df[target_col] == cls, feature_col].dropna()
        if len(series) > 0:
            groups.append(series.values)
            labels.append(str(cls))

    if len(groups) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    ax.boxplot(groups, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(target_col)
    ax.set_ylabel(ylabel)

# %% [markdown]
# ## 1) Repository inspection and data discovery
#
# This section checks the current repository structure and confirms which data files are available locally.
#
# The notebook is designed to work whether you run it from:
#
# - the repository root, or
# - a `notebooks/` subfolder inside the repository

# %%
repo_root = locate_repo_root()
data_dir = repo_root / "raw_data"

EXPECTED_FILES = {
    "train_transaction": ["train_transaction.csv.gz", "train_transaction.csv"],
    "train_identity": ["train_identity.csv.gz", "train_identity.csv"],
    "test_transaction": ["test_transaction.csv.gz", "test_transaction.csv"],
    "test_identity": ["test_identity.csv.gz", "test_identity.csv"],
    "sample_submission": ["sample_submission.csv.gz", "sample_submission.csv"],
}

paths = {
    name: resolve_existing_path(data_dir, candidates)
    for name, candidates in EXPECTED_FILES.items()
}

repo_file_rows = []
for path in sorted(data_dir.glob("*")):
    if path.is_file():
        repo_file_rows.append(
            {
                "file_name": path.name,
                "suffixes": "".join(path.suffixes),
                "size_bytes": path.stat().st_size,
                "size_human": human_size(path.stat().st_size),
            }
        )

repo_files_df = pd.DataFrame(repo_file_rows).sort_values("file_name").reset_index(drop=True)

display(Markdown(f"**Detected repo root:** `{repo_root}`"))
display(Markdown(f"**Detected data directory:** `{data_dir}`"))
display(repo_files_df)

missing_expected = [name for name, path in paths.items() if path is None]
if missing_expected:
    print("Missing expected files:", missing_expected)
else:
    print("All expected raw data files were detected.")

print("\nResolved file paths:")
for name, path in paths.items():
    print(f"- {name}: {path}")

# %% [markdown]
# ### Dataset choice for this EDA
#
# Given the current repo structure, this notebook will use:
#
# - **Primary labeled file:** `train_transaction`
# - **Auxiliary file:** `train_identity`
# - **Optional comparison file:** `test_transaction` (only `TransactionDT` column, if available)
#
# That is the most appropriate choice for fraud EDA because the target label should live in `train_transaction`.

# %% [markdown]
# ## 2) Setup + load dataset
#
# This section loads:
#
# - `train_transaction`
# - `train_identity`
#
# It also performs lightweight memory optimization so the notebook remains practical on a student machine.

# %%
required_files = ["train_transaction", "train_identity"]
for required_name in required_files:
    if paths[required_name] is None:
        raise FileNotFoundError(
            f"Required file '{required_name}' was not found in {data_dir}. "
            f"Please check the file path or upload the missing file."
        )

train_tx_path = paths["train_transaction"]
train_id_path = paths["train_identity"]

train_tx_compression = "gzip" if "".join(train_tx_path.suffixes).endswith(".gz") else None
train_id_compression = "gzip" if "".join(train_id_path.suffixes).endswith(".gz") else None

print("Loading train_transaction...")
train_transaction = pd.read_csv(train_tx_path, compression=train_tx_compression, low_memory=False)
train_transaction = reduce_mem_usage(train_transaction)

print("Loading train_identity...")
train_identity = pd.read_csv(train_id_path, compression=train_id_compression, low_memory=False)
train_identity = reduce_mem_usage(train_identity)

target_col = detect_target_column(train_transaction)
if target_col is None:
    raise ValueError(
        "Could not detect the target column automatically. "
        "Please edit TARGET_CANDIDATES or set target_col manually."
    )

if "TransactionID" in train_transaction.columns and "TransactionID" in train_identity.columns:
    identity_id_set = set(train_identity["TransactionID"].astype("int64").tolist())
    train_transaction["has_identity"] = train_transaction["TransactionID"].astype("int64").isin(identity_id_set).astype("int8")
else:
    train_transaction["has_identity"] = 0

train_numeric_cols, train_categorical_cols = make_feature_lists(train_transaction, target_col=target_col)
identity_numeric_cols, identity_categorical_cols = make_feature_lists(train_identity, target_col=None)

print(f"Target column detected: {target_col}")
print(f"train_transaction shape: {train_transaction.shape}")
print(f"train_identity shape: {train_identity.shape}")
print(f"Numeric features in train_transaction (excluding target): {len(train_numeric_cols)}")
print(f"Categorical features in train_transaction: {len(train_categorical_cols)}")

# %% [markdown]
# ## 3) Dataset overview
#
# This section validates the basic structure of the dataset:
#
# - shape
# - head
# - columns
# - data types
# - target availability
# - identity coverage

# %%
overview_df = pd.DataFrame(
    {
        "table": ["train_transaction", "train_identity"],
        "rows": [train_transaction.shape[0], train_identity.shape[0]],
        "columns": [train_transaction.shape[1], train_identity.shape[1]],
        "memory_mb_approx": [
            round(train_transaction.memory_usage(deep=True).sum() / 1024**2, 2),
            round(train_identity.memory_usage(deep=True).sum() / 1024**2, 2),
        ],
    }
)

display(overview_df)

display(Markdown("### `train_transaction` head"))
display(train_transaction.head())

display(Markdown("### `train_identity` head"))
display(train_identity.head())

dtype_summary_tx = (
    train_transaction.dtypes.astype(str)
    .value_counts()
    .rename_axis("dtype")
    .reset_index(name="count")
)
display(Markdown("### Dtype summary for `train_transaction`"))
display(dtype_summary_tx)

column_overview = pd.DataFrame(
    {
        "total_columns": [train_transaction.shape[1]],
        "numeric_columns_ex_target": [len(train_numeric_cols)],
        "categorical_columns": [len(train_categorical_cols)],
        "target_column": [target_col],
        "target_unique_values": [train_transaction[target_col].nunique(dropna=False)],
    }
)
display(Markdown("### Column overview"))
display(column_overview)

if "TransactionID" in train_transaction.columns:
    transaction_id_dup = train_transaction["TransactionID"].duplicated().sum()
    print(f"Duplicate TransactionID count in train_transaction: {transaction_id_dup:,}")

if "TransactionID" in train_identity.columns:
    identity_id_dup = train_identity["TransactionID"].duplicated().sum()
    print(f"Duplicate TransactionID count in train_identity: {identity_id_dup:,}")

if "has_identity" in train_transaction.columns:
    identity_coverage = train_transaction["has_identity"].mean()
    print(f"Identity coverage in train_transaction: {identity_coverage:.2%}")

# %% [markdown]
# ## 4) Target analysis
#
# Fraud detection datasets are usually dominated by **non-fraud** transactions.
#
# This section measures:
#
# - class counts
# - fraud ratio
# - imbalance severity
# - why this matters for Recall, Precision, F1-score, and PR-AUC

# %%
class_counts = train_transaction[target_col].value_counts(dropna=False).sort_index()
class_ratios = class_counts / len(train_transaction)

class_dist_df = pd.DataFrame(
    {
        "class_value": class_counts.index,
        "count": class_counts.values,
        "ratio": class_ratios.values,
    }
)
class_dist_df["class_label"] = class_dist_df["class_value"].map({0: "non-fraud", 1: "fraud"}).fillna(
    class_dist_df["class_value"].astype(str)
)

display(class_dist_df)

positive_rate = float(class_ratios.get(1, np.nan))
negative_rate = float(class_ratios.get(0, np.nan))
imbalance_ratio = class_counts.get(0, 0) / max(class_counts.get(1, 1), 1)

print(f"Fraud rate: {positive_rate:.4%}")
print(f"Non-fraud rate: {negative_rate:.4%}")
print(f"Imbalance ratio (non-fraud : fraud) ≈ {imbalance_ratio:.1f} : 1")
print(f"Random-classifier PR-AUC baseline ≈ positive class prevalence = {positive_rate:.6f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(class_dist_df["class_label"].astype(str), class_dist_df["count"])

for i, row in class_dist_df.iterrows():
    ax.text(
        i,
        row["count"],
        f'{row["count"]:,}\n({row["ratio"]:.2%})',
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_title("Class distribution: fraud vs non-fraud")
ax.set_xlabel("")
ax.set_ylabel("Number of transactions")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why this matters for modeling
#
# In fraud detection, heavy class imbalance directly affects model behavior:
#
# - A model can get high **accuracy** while still missing most frauds.
# - **Recall** matters because false negatives are costly.
# - **Precision** matters because too many false positives create operational noise.
# - **F1-score** is useful once you choose a threshold.
# - **PR-AUC** is usually more informative than accuracy for imbalanced fraud data.
#
# A strong EDA should therefore look for features and data patterns that help separate fraud from non-fraud, especially under severe imbalance.

# %% [markdown]
# ## 5) Basic data inspection (EDA-level)
#
# This section checks:
#
# - missing values
# - duplicate rows
# - duplicate IDs
# - constant or near-constant columns
# - obvious abnormal values
# - numerical and categorical summaries

# %%
tx_missing = missing_summary(train_transaction)
id_missing = missing_summary(train_identity)

display(Markdown("### Top missing columns in `train_transaction`"))
display(tx_missing.head(TOP_N_MISSING))

display(Markdown("### Top missing columns in `train_identity`"))
display(id_missing.head(TOP_N_MISSING))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

tx_plot = tx_missing.head(TOP_N_MISSING).sort_values("missing_pct", ascending=True)
axes[0].barh(tx_plot["column"].astype(str), tx_plot["missing_pct"])
axes[0].set_title("Top missing columns in train_transaction")
axes[0].set_xlabel("Missing percentage")
axes[0].set_ylabel("")

id_plot = id_missing.head(TOP_N_MISSING).sort_values("missing_pct", ascending=True)
axes[1].barh(id_plot["column"].astype(str), id_plot["missing_pct"])
axes[1].set_title("Top missing columns in train_identity")
axes[1].set_xlabel("Missing percentage")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# %%
duplicate_tx_rows = int(train_transaction.duplicated().sum())
duplicate_id_rows = int(train_identity.duplicated().sum())

constant_cols_tx = [col for col in train_transaction.columns if train_transaction[col].nunique(dropna=False) <= 1]
constant_cols_id = [col for col in train_identity.columns if train_identity[col].nunique(dropna=False) <= 1]

dominant_value_rows = []
for col in train_transaction.columns:
    top_share = train_transaction[col].value_counts(dropna=False, normalize=True)
    if not top_share.empty:
        top_value_share = float(top_share.iloc[0])
        if top_value_share >= 0.95:
            dominant_value_rows.append({"column": col, "top_value_share": top_value_share})

dominant_value_df = (
    pd.DataFrame(dominant_value_rows)
    .sort_values("top_value_share", ascending=False)
    .reset_index(drop=True)
    if dominant_value_rows
    else pd.DataFrame(columns=["column", "top_value_share"])
)

basic_checks = {
    "duplicate_rows_train_transaction": duplicate_tx_rows,
    "duplicate_rows_train_identity": duplicate_id_rows,
    "constant_cols_train_transaction": len(constant_cols_tx),
    "constant_cols_train_identity": len(constant_cols_id),
}

if "TransactionID" in train_transaction.columns:
    basic_checks["duplicate_transaction_ids_train_transaction"] = int(train_transaction["TransactionID"].duplicated().sum())
if "TransactionID" in train_identity.columns:
    basic_checks["duplicate_transaction_ids_train_identity"] = int(train_identity["TransactionID"].duplicated().sum())

if "TransactionAmt" in train_transaction.columns:
    basic_checks["TransactionAmt_le_zero_count"] = int((train_transaction["TransactionAmt"] <= 0).sum())
    basic_checks["TransactionAmt_missing_count"] = int(train_transaction["TransactionAmt"].isna().sum())

display(pd.DataFrame([basic_checks]).T.rename(columns={0: "value"}))

display(Markdown("### Constant columns in `train_transaction`"))
display(pd.DataFrame({"constant_column": constant_cols_tx}).head(50))

display(Markdown("### Near-constant columns in `train_transaction` (top value share >= 95%)"))
display(dominant_value_df.head(30))

numeric_description = train_transaction[train_numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
display(Markdown("### Numerical feature summary (`train_transaction`)"))
display(numeric_description.head(40))

if train_categorical_cols:
    categorical_summary_rows = []
    for col in train_categorical_cols:
        vc = train_transaction[col].value_counts(dropna=False, normalize=True)
        top_value = vc.index[0] if len(vc) > 0 else np.nan
        top_share = vc.iloc[0] if len(vc) > 0 else np.nan
        categorical_summary_rows.append(
            {
                "column": col,
                "nunique_including_nan": train_transaction[col].nunique(dropna=False),
                "missing_pct": train_transaction[col].isna().mean(),
                "top_value": str(top_value),
                "top_value_share": top_share,
            }
        )

    categorical_summary_df = pd.DataFrame(categorical_summary_rows).sort_values(
        ["missing_pct", "nunique_including_nan"], ascending=[False, False]
    )
    display(Markdown("### Categorical feature summary (`train_transaction`)"))
    display(categorical_summary_df.head(30))

# %% [markdown]
# ### Missingness itself may be predictive
#
# In fraud detection, high-missing columns are not automatically useless.
#
# Sometimes:
#
# - missingness is caused by a different user flow,
# - identity data exists for only part of the traffic,
# - sparse features may correlate with fraud,
# - a **missing indicator** later becomes more useful than raw imputation.
#
# So the next check compares missingness patterns between fraud and non-fraud.

# %%
missing_signal_rows = []
fraud_mask = train_transaction[target_col] == 1
nonfraud_mask = train_transaction[target_col] == 0

for col in train_transaction.columns:
    if col in {target_col, "TransactionID"}:
        continue

    is_missing = train_transaction[col].isna()
    missing_pct = is_missing.mean()

    if missing_pct == 0:
        continue

    fraud_missing_pct = is_missing[fraud_mask].mean()
    nonfraud_missing_pct = is_missing[nonfraud_mask].mean()

    fraud_rate_if_missing = train_transaction.loc[is_missing, target_col].mean()
    fraud_rate_if_present = train_transaction.loc[~is_missing, target_col].mean()

    missing_signal_rows.append(
        {
            "column": col,
            "overall_missing_pct": missing_pct,
            "fraud_missing_pct": fraud_missing_pct,
            "nonfraud_missing_pct": nonfraud_missing_pct,
            "missing_gap": fraud_missing_pct - nonfraud_missing_pct,
            "abs_missing_gap": abs(fraud_missing_pct - nonfraud_missing_pct),
            "fraud_rate_if_missing": fraud_rate_if_missing,
            "fraud_rate_if_present": fraud_rate_if_present,
        }
    )

missing_signal_df = (
    pd.DataFrame(missing_signal_rows)
    .sort_values("abs_missing_gap", ascending=False)
    .reset_index(drop=True)
)

display(Markdown("### Columns where missingness differs most between fraud and non-fraud"))
display(missing_signal_df.head(20))

fig, ax = plt.subplots(figsize=(10, 7))
plot_df = missing_signal_df.head(15).sort_values("abs_missing_gap", ascending=True)
ax.barh(plot_df["column"].astype(str), plot_df["abs_missing_gap"])
ax.set_title("Top columns by missingness gap between classes")
ax.set_xlabel("Absolute gap in missing percentage")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6) Univariate analysis
#
# Because this dataset has many columns, plotting every feature would be noisy and slow.
#
# Instead, this notebook selects a practical subset of numerical variables based on:
#
# - known transaction-relevant fields (for example `TransactionAmt`, `TransactionDT`, `dist*`, `C*`, `D*` if present)
# - strongest association with the target
# - sufficient non-null coverage
#
# This keeps the notebook practical while still surfacing useful fraud patterns.

# %%
eda_sample = stratified_sample(train_transaction, target_col=target_col, max_rows=MAX_PLOT_ROWS, random_state=RANDOM_STATE)
univariate_features = choose_numeric_features(train_transaction, target_col=target_col, n=12)

print(f"EDA sample size used for heavy plots: {len(eda_sample):,}")
print("Selected numerical features for univariate plots:")
print(univariate_features)

# %%
if not univariate_features:
    print("No suitable numerical features were selected for univariate analysis.")
else:
    n_cols = 2
    n_rows = math.ceil(len(univariate_features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, univariate_features):
        feature_sample = eda_sample[col].dropna()
        ax.hist(feature_sample, bins=50)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    for ax in axes[len(univariate_features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# %%
if univariate_features:
    univariate_summary = []
    for col in univariate_features:
        s = train_transaction[col]
        univariate_summary.append(
            {
                "feature": col,
                "non_null_ratio": s.notna().mean(),
                "nunique": s.nunique(dropna=True),
                "mean": s.mean(),
                "std": s.std(),
                "skew": s.skew(),
                "q01": s.quantile(0.01),
                "q50": s.quantile(0.50),
                "q99": s.quantile(0.99),
            }
        )

    univariate_summary_df = pd.DataFrame(univariate_summary).sort_values(
        "skew", key=lambda x: x.abs(), ascending=False
    )
    display(univariate_summary_df)

if "TransactionAmt" in train_transaction.columns:
    amt_sample = train_transaction["TransactionAmt"].dropna()
    amt_sample = amt_sample.sample(n=min(len(amt_sample), MAX_PLOT_ROWS), random_state=RANDOM_STATE)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].hist(amt_sample, bins=60)
    axes[0].set_title("TransactionAmt distribution")
    axes[0].set_xlabel("TransactionAmt")
    axes[0].set_ylabel("Frequency")

    log_amt = np.log1p(amt_sample.clip(lower=0))
    axes[1].hist(log_amt, bins=60)
    axes[1].set_title("log1p(TransactionAmt) distribution")
    axes[1].set_xlabel("log1p(TransactionAmt)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    amt_quantile_table = quantile_fraud_table(train_transaction, "TransactionAmt", target_col, q=10)
    if amt_quantile_table is not None:
        display(Markdown("### Fraud rate by TransactionAmt decile"))
        display(amt_quantile_table)

# %% [markdown]
# ### Practical interpretation notes for univariate analysis
#
# When reading the plots above, pay special attention to:
#
# - **strong skewness**: many fraud-related features are highly skewed
# - **heavy tails**: large-value tails may contain disproportionate fraud risk
# - **sparsity**: features with many missing values may still be useful
# - **non-Gaussian shape**: tree-based models often handle these distributions better than linear assumptions
#
# These observations matter later when choosing:
# - scaling strategy,
# - sampling strategy,
# - threshold tuning,
# - and model family.

# %% [markdown]
# ## 7) Fraud vs Non-Fraud comparison
#
# The goal here is to identify variables whose distributions differ between the two classes.
#
# This is one of the most important EDA steps for fraud detection because it helps answer:
#
# - which features might separate fraud from non-fraud,
# - where fraud tends to concentrate,
# - which missingness or category slices could help Recall and PR-AUC later.

# %%
categorical_interest = [
    col for col in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "has_identity"]
    if col in train_transaction.columns
]

if categorical_interest:
    plot_features = [col for col in ["ProductCD", "card4", "card6", "has_identity"] if col in categorical_interest]
    n_cols = 2
    n_rows = math.ceil(len(plot_features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, plot_features):
        profile = category_profile(train_transaction, feature, target_col, min_count=200, top_n=15, sort_by="fraud_rate")
        x_labels = profile[feature].astype(str).tolist()
        ax.bar(x_labels, profile["fraud_rate"])
        ax.set_title(f"Fraud rate by {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Fraud rate")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(plot_features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

for feature in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "has_identity"]:
    if feature in train_transaction.columns:
        display(Markdown(f"### Category profile for `{feature}`"))
        display(category_profile(train_transaction, feature, target_col, min_count=200, top_n=15, sort_by="fraud_rate"))

if {"TransactionID", target_col}.issubset(set(train_transaction.columns)) and {"TransactionID"}.issubset(set(train_identity.columns)):
    identity_cols_to_analyze = [col for col in ["DeviceType", "DeviceInfo"] if col in train_identity.columns]
    if identity_cols_to_analyze:
        joined_identity_view = train_transaction[["TransactionID", target_col]].merge(
            train_identity[["TransactionID"] + identity_cols_to_analyze],
            on="TransactionID",
            how="left",
        )

        for feature in identity_cols_to_analyze:
            display(Markdown(f"### Identity-side category profile for `{feature}`"))
            display(category_profile(joined_identity_view, feature, target_col, min_count=200, top_n=15, sort_by="fraud_rate"))

# %%
top_diff_features = top_numeric_by_target_corr(
    train_transaction,
    target_col=target_col,
    top_n=6,
    min_non_null_ratio=0.20,
    exclude={"TransactionID"},
).index.tolist()

comparison_df = train_transaction[[target_col] + top_diff_features].copy()
comparison_sample = balanced_class_sample(
    comparison_df.dropna(how="all", subset=top_diff_features),
    target_col=target_col,
    max_per_class=MAX_COMPARE_PER_CLASS,
    random_state=RANDOM_STATE,
)

print("Top differentiating numerical features (by |correlation with target|):")
print(top_diff_features)
print(f"Balanced sample used for class-comparison plots: {len(comparison_sample):,} rows")

# %%
if top_diff_features:
    n_cols = 2
    n_rows = math.ceil(len(top_diff_features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, top_diff_features):
        plot_df = comparison_sample[[target_col, feature]].dropna().copy()

        if plot_df.empty:
            ax.axis("off")
            continue

        low, high = plot_df[feature].quantile([0.01, 0.99])
        plot_df["clipped_feature"] = plot_df[feature].clip(lower=low, upper=high)

        class_boxplot(
            ax=ax,
            df=plot_df,
            feature_col="clipped_feature",
            target_col=target_col,
            title=f"{feature} by class (clipped to 1st–99th pct for display)",
            ylabel=feature,
        )

    for ax in axes[len(top_diff_features):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

comparison_summary_rows = []
for feature in top_diff_features:
    temp = train_transaction[[feature, target_col]].dropna()
    if temp.empty:
        continue

    nonfraud_series = temp.loc[temp[target_col] == 0, feature]
    fraud_series = temp.loc[temp[target_col] == 1, feature]

    comparison_summary_rows.append(
        {
            "feature": feature,
            "missing_pct": train_transaction[feature].isna().mean(),
            "nonfraud_mean": nonfraud_series.mean(),
            "fraud_mean": fraud_series.mean(),
            "nonfraud_median": nonfraud_series.median(),
            "fraud_median": fraud_series.median(),
            "mean_delta": fraud_series.mean() - nonfraud_series.mean(),
            "median_delta": fraud_series.median() - nonfraud_series.median(),
        }
    )

comparison_summary_df = pd.DataFrame(comparison_summary_rows).sort_values(
    "mean_delta", key=lambda x: x.abs(), ascending=False
)
display(comparison_summary_df)

# %% [markdown]
# ### Practical interpretation notes for fraud vs non-fraud comparison
#
# When a feature shows clear class separation, it can help later with:
#
# - ranking suspicious transactions,
# - threshold selection,
# - improving Recall without collapsing Precision,
# - and increasing PR-AUC.
#
# But be careful:
#
# - separation in EDA does **not** guarantee production value,
# - some signals may be unstable over time,
# - some large differences may come from missingness or drift rather than true business signal.
#
# That is why time-aware validation will matter later.

# %% [markdown]
# ## 8) Correlation analysis
#
# This section looks at:
#
# - which numerical features are most correlated with the target
# - correlation structure among top candidate features
# - highly correlated pairs that may indicate redundancy
#
# Because the dataset has many columns, a full heatmap would be unreadable.
# So this notebook focuses on the most relevant numerical subset.

# %%
target_corr = top_numeric_by_target_corr(
    train_transaction,
    target_col=target_col,
    top_n=TOP_N_CORR,
    min_non_null_ratio=0.20,
    exclude={"TransactionID"},
)

target_corr_df = target_corr.rename("abs_corr_with_target").reset_index().rename(columns={"index": "feature"})
display(Markdown("### Top numerical features by absolute correlation with target"))
display(target_corr_df)

heatmap_features = target_corr.index[:15].tolist()
if "TransactionAmt" in train_transaction.columns and "TransactionAmt" not in heatmap_features:
    heatmap_features = ["TransactionAmt"] + heatmap_features[:-1]

corr_matrix = train_transaction[heatmap_features + [target_col]].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_matrix.values, aspect="auto")
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticks(range(len(corr_matrix.index)))
ax.set_yticklabels(corr_matrix.index)
ax.set_title("Correlation heatmap for top target-related numerical features")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

pair_feature_pool = target_corr.index[:30].tolist()
high_corr_df = high_corr_pairs(train_transaction[pair_feature_pool], threshold=0.85)

display(Markdown("### Highly correlated feature pairs among the selected pool"))
display(high_corr_df.head(30))

# %% [markdown]
# ### Correlation interpretation guidance
#
# In this dataset, strong feature-feature correlation can matter because:
#
# - it may signal redundant engineered variables,
# - it may create instability for linear models,
# - it may be harmless for tree ensembles but still affect interpretability,
# - it can influence feature selection and regularization choices later.
#
# For fraud detection, correlation with the target is useful as a **screening signal**, but not the final truth.
# Nonlinear interactions and missingness effects often matter a lot more.

# %% [markdown]
# ## 9) Outlier analysis
#
# Outliers are especially important in fraud detection.
#
# Very important caution:
#
# - an outlier is **not automatically bad data**
# - in fraud detection, outliers may be the signal itself
# - blindly removing them can reduce Recall and damage PR-AUC
#
# So the goal here is **not** to remove outliers yet.
# The goal is to measure where outliers occur and whether outlier regions have higher fraud concentration.

# %%
outlier_features = []
for feature in ["TransactionAmt", "dist1", "dist2"] + top_diff_features:
    if feature in train_transaction.columns and feature not in outlier_features:
        outlier_features.append(feature)

outlier_summary_df = iqr_outlier_summary(train_transaction, outlier_features[:8], target_col=target_col)

display(outlier_summary_df)

if not outlier_summary_df.empty:
    plot_features = outlier_summary_df["feature"].head(4).tolist()
    fig, axes = plt.subplots(len(plot_features), 1, figsize=(14, 4.5 * len(plot_features)))
    if len(plot_features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, plot_features):
        plot_df = balanced_class_sample(
            train_transaction[[target_col, feature]].dropna(),
            target_col=target_col,
            max_per_class=min(MAX_COMPARE_PER_CLASS, 10_000),
            random_state=RANDOM_STATE,
        )

        low, high = plot_df[feature].quantile([0.01, 0.99])
        plot_df["clipped_feature"] = plot_df[feature].clip(lower=low, upper=high)

        class_boxplot(
            ax=ax,
            df=plot_df,
            feature_col="clipped_feature",
            target_col=target_col,
            title=f"{feature}: class-wise view for outlier-sensitive behavior",
            ylabel=feature,
        )

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Outlier interpretation
#
# If the fraud rate among outliers is much higher than the fraud rate among inliers, that is an important modeling clue.
#
# Possible later actions for modeling:
#
# - keep raw values for tree-based models,
# - test log transforms for highly skewed positive variables,
# - consider robust scaling for linear models,
# - avoid blindly clipping outliers before validation.
#
# For fraud detection, it is often better to **model the unusual pattern** than to discard it.

# %% [markdown]
# ## 10) Time-based analysis
#
# If the dataset contains a time-like column, fraud EDA should always inspect temporal behavior.
#
# Why this matters:
#
# - fraud patterns often drift over time
# - random train/validation splits can overestimate performance
# - time structure may affect Recall, Precision, F1, and PR-AUC in real deployment
#
# For the common IEEE-CIS fraud dataset format, `TransactionDT` is usually a **relative elapsed-time column**, not a human-readable timestamp.
# So the analysis below uses **relative day / week / hour** instead of calendar dates.

# %%
if "TransactionDT" not in train_transaction.columns:
    print("No TransactionDT column found. Skipping time-based analysis.")
else:
    time_df = train_transaction[[target_col, "TransactionDT"]].copy()
    time_df["transaction_day"] = (time_df["TransactionDT"] // 86400).astype(int)
    time_df["transaction_week"] = (time_df["transaction_day"] // 7).astype(int)
    time_df["transaction_hour"] = ((time_df["TransactionDT"] % 86400) // 3600).astype(int)
    time_df["day_index_mod7"] = (time_df["transaction_day"] % 7).astype(int)

    daily_volume = time_df.groupby("transaction_day").size().rename("transaction_count").reset_index()
    daily_fraud = time_df.groupby("transaction_day")[target_col].sum().rename("fraud_count").reset_index()
    daily_rate = time_df.groupby("transaction_day")[target_col].mean().rename("fraud_rate").reset_index()
    hourly_rate = time_df.groupby("transaction_hour")[target_col].mean().rename("fraud_rate").reset_index()
    weekly_rate = time_df.groupby("transaction_week")[target_col].mean().rename("fraud_rate").reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(daily_volume["transaction_day"], daily_volume["transaction_count"])
    axes[0, 0].set_title("Transaction volume by relative day")
    axes[0, 0].set_xlabel("transaction_day")
    axes[0, 0].set_ylabel("transaction_count")

    axes[0, 1].plot(daily_fraud["transaction_day"], daily_fraud["fraud_count"])
    axes[0, 1].set_title("Fraud count by relative day")
    axes[0, 1].set_xlabel("transaction_day")
    axes[0, 1].set_ylabel("fraud_count")

    axes[1, 0].plot(daily_rate["transaction_day"], daily_rate["fraud_rate"])
    axes[1, 0].set_title("Fraud rate by relative day")
    axes[1, 0].set_xlabel("transaction_day")
    axes[1, 0].set_ylabel("fraud_rate")

    axes[1, 1].bar(hourly_rate["transaction_hour"].astype(str), hourly_rate["fraud_rate"])
    axes[1, 1].set_title("Fraud rate by relative hour")
    axes[1, 1].set_xlabel("transaction_hour")
    axes[1, 1].set_ylabel("fraud_rate")

    plt.tight_layout()
    plt.show()

    display(Markdown("### Weekly fraud-rate summary"))
    display(weekly_rate.head(20))

    if paths["test_transaction"] is not None:
        test_tx_path = paths["test_transaction"]
        test_tx_compression = "gzip" if "".join(test_tx_path.suffixes).endswith(".gz") else None

        try:
            test_time = pd.read_csv(
                test_tx_path,
                usecols=["TransactionDT"],
                compression=test_tx_compression,
                low_memory=False,
            )

            split_time_summary = pd.DataFrame(
                {
                    "split": ["train", "test"],
                    "min_TransactionDT": [
                        train_transaction["TransactionDT"].min(),
                        test_time["TransactionDT"].min(),
                    ],
                    "max_TransactionDT": [
                        train_transaction["TransactionDT"].max(),
                        test_time["TransactionDT"].max(),
                    ],
                }
            )
            display(Markdown("### Train vs test time range"))
            display(split_time_summary)

            train_time_sample = train_transaction["TransactionDT"].dropna()
            test_time_sample = test_time["TransactionDT"].dropna()

            train_time_sample = train_time_sample.sample(
                n=min(len(train_time_sample), 100_000),
                random_state=RANDOM_STATE,
            )
            test_time_sample = test_time_sample.sample(
                n=min(len(test_time_sample), 100_000),
                random_state=RANDOM_STATE,
            )

            bins = np.linspace(
                min(train_time_sample.min(), test_time_sample.min()),
                max(train_time_sample.max(), test_time_sample.max()),
                60,
            )

            plt.figure(figsize=(12, 5))
            plt.hist(train_time_sample, bins=bins, histtype="step", density=True, label="train")
            plt.hist(test_time_sample, bins=bins, histtype="step", density=True, label="test")
            plt.title("Train vs test TransactionDT distribution")
            plt.xlabel("TransactionDT")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as exc:
            print("Could not load test_transaction for time-range comparison.")
            print(f"Reason: {exc}")

# %% [markdown]
# ### Time-based modeling implication
#
# If train and test occupy different time ranges, or fraud rate changes across relative day/week/hour, then:
#
# - random validation splits may be overly optimistic,
# - time-based validation should be tested later,
# - threshold tuning may need to be re-checked under drift,
# - a model optimized only on random CV may fail on PR-AUC or Recall in deployment-like conditions.

# %% [markdown]
# ## 11) EDA summary
#
# This section collects the main findings from the notebook and translates them into modeling implications.

# %%
summary_lines = []

summary_lines.append(
    f"- **Class imbalance is severe**: fraud rate = **{positive_rate:.4%}** "
    f"({class_counts.get(1, 0):,}/{len(train_transaction):,}). "
    "This makes **PR-AUC, Recall, Precision, and F1-score** much more informative than accuracy."
)

if "has_identity" in train_transaction.columns:
    summary_lines.append(
        f"- **Identity coverage**: **{train_transaction['has_identity'].mean():.2%}** of training transactions "
        "have a matching identity row. Identity presence itself may carry useful signal."
    )

if not tx_missing.empty:
    top_missing_cols = ", ".join(tx_missing.head(5)["column"].tolist())
    summary_lines.append(
        f"- **High missingness is a major property of the dataset**. Top missing columns include: {top_missing_cols}. "
        "Do not drop missing-heavy features blindly; missingness may itself be predictive."
    )

if not missing_signal_df.empty:
    top_missing_signal_cols = ", ".join(missing_signal_df.head(5)["column"].tolist())
    summary_lines.append(
        f"- **Missingness differs by class** for several columns, especially: {top_missing_signal_cols}. "
        "That is a strong sign to test missing indicators in later modeling."
    )

if not target_corr_df.empty:
    top_corr_cols = ", ".join(target_corr_df.head(5)["feature"].tolist())
    summary_lines.append(
        f"- **Numerical features with the strongest target association** in this EDA include: {top_corr_cols}. "
        "These are high-priority candidates for deeper feature-level validation."
    )

if not outlier_summary_df.empty:
    outlier_cols = ", ".join(outlier_summary_df.head(5)["feature"].tolist())
    summary_lines.append(
        f"- **Outlier-heavy variables** such as {outlier_cols} should not be removed automatically. "
        "In fraud detection, extreme values may improve Recall if handled correctly."
    )

if "TransactionDT" in train_transaction.columns:
    summary_lines.append(
        "- **Time structure exists** in the dataset through `TransactionDT`. "
        "Use time-aware validation later, because fraud rate and feature behavior may drift."
    )

summary_lines.append(
    "- **Modeling implication**: later experiments should prioritize class-imbalance-aware training, "
    "threshold tuning, PR-AUC monitoring, Recall/Precision trade-off analysis, and validation splits that respect time."
)

display(Markdown("### Key EDA takeaways\n" + "\n".join(summary_lines)))

# %% [markdown]
# # 12) DVC evaluation for the current repo
#
# This section does **not** discuss DVC in theory only.
#
# It evaluates DVC using the current repository situation:
#
# - raw dataset files already exist inside `raw_data/`
# - the repo appears to be at an early stage
# - you are currently focused on EDA
# - the next likely stages are preprocessing, feature engineering, training, and experiment iteration

# %%
raw_total_bytes = int(repo_files_df["size_bytes"].sum()) if not repo_files_df.empty else 0
raw_total_mb = raw_total_bytes / 1024**2

dvc_status_df = pd.DataFrame(
    {
        "check": [
            "raw_data folder exists",
            "raw_data total size (MB)",
            ".dvc directory exists",
            "dvc.yaml exists",
            "DVC available in environment",
        ],
        "value": [
            data_dir.exists(),
            round(raw_total_mb, 2),
            (repo_root / ".dvc").exists(),
            (repo_root / "dvc.yaml").exists(),
            get_dvc_version() if get_dvc_version() is not None else "Not installed",
        ],
    }
)

display(dvc_status_df)

tracking_recommendation_rows = []
for _, row in repo_files_df.iterrows():
    file_name = row["file_name"]
    recommended_tracking = "DVC" if file_name.endswith((".csv", ".csv.gz", ".parquet", ".pkl", ".joblib")) else "Git"
    tracking_recommendation_rows.append(
        {
            "file_name": file_name,
            "size_human": row["size_human"],
            "recommended_tracking": recommended_tracking,
        }
    )

display(Markdown("### File-level tracking recommendation"))
display(pd.DataFrame(tracking_recommendation_rows))

# %% [markdown]
# ## DVC decision for this repo
#
# ### Final decision: **2. Nên làm EDA trước, DVC sau**
#
# That is the most practical decision for your current project state.
#
# ### Why this is the right decision here
#
# 1. **Your immediate goal is EDA**, and you already have the raw data in the repository structure needed to start.
# 2. The repo appears **very early-stage**, so the biggest short-term value comes from understanding the data first.
# 3. **DVC becomes much more valuable the moment you start creating new large artifacts**, such as:
#    - merged train datasets,
#    - processed feature tables,
#    - sampled training sets,
#    - model binaries,
#    - validation predictions,
#    - experiment outputs.
# 4. If you postpone DVC for too long and start committing generated datasets/models into Git, the repo will become messy quickly.
#
# ### So the practical interpretation is:
#
# - **Do the EDA now**
# - then **introduce DVC immediately before preprocessing / feature engineering / training artifacts start to accumulate**
#
# That is why the recommendation is **not** “ignore DVC”.
# It is: **EDA first, then integrate DVC very soon after this notebook**.

# %% [markdown]
# ## What DVC would help with in this fraud-detection project
#
# For your specific project, DVC would help in a very practical way:
#
# - version the raw dataset without bloating Git further
# - track generated data such as merged transaction + identity tables
# - track processed datasets used for modeling
# - track larger model artifacts and prediction files
# - let teammates pull the exact same data version
# - keep Git focused on code / notebooks / config, while DVC manages data lineage
#
# This is especially useful once you begin comparing experiments aimed at improving:
#
# - Recall
# - Precision
# - F1-score
# - PR-AUC

# %% [markdown]
# ## What should go into Git vs DVC
#
# ### Put in normal Git
# - `README.md`
# - notebooks (`.ipynb`)
# - Python scripts under `src/`
# - `requirements.txt` / `environment.yml`
# - small config files
# - `.dvc/`
# - `dvc.yaml`
# - `dvc.lock`
# - `params.yaml`
# - `*.dvc` metadata files
# - small reports / figures / markdown docs
#
# ### Put in DVC
# - `raw_data/` (or later `data/raw/`)
# - merged datasets
# - processed datasets
# - sampled training datasets
# - feature matrices
# - large prediction outputs
# - trained model artifacts (`.pkl`, `.joblib`, `.onnx`, etc.) when they become non-trivial in size
#
# ### Do not commit to either Git history casually
# - `.venv/`
# - `__pycache__/`
# - `.ipynb_checkpoints/`
# - local logs
# - temporary scratch outputs
# - one-off debug dumps

# %% [markdown]
# ## Minimal DVC rollout plan that fits the current repo
#
# Because your repo already uses `raw_data/`, the least disruptive DVC adoption path is:
#
# - keep the current `raw_data/` directory name for now
# - start tracking that directory with DVC
# - only reorganize into `data/raw/` later if the project grows
#
# ### Step 1 — Install DVC
#
# If you only need local storage first:
#
# ```bash
# pip install dvc
# ```
#
# If your team wants a Google Drive remote:
#
# ```bash
# pip install "dvc[gdrive]"
# ```
#
# ### Step 2 — Initialize DVC
#
# Run from the repository root:
#
# ```bash
# dvc init
# git add .dvc .dvcignore
# git commit -m "Initialize DVC"
# ```
#
# ### Step 3 — Stop tracking raw data with Git, then add it to DVC
#
# ```bash
# git rm -r --cached raw_data
# dvc add raw_data
# git add raw_data.dvc .gitignore
# git commit -m "Track raw_data with DVC"
# ```
#
# ### Important practical warning
#
# If `raw_data/` was already committed into Git history earlier, then **adding DVC now does not automatically shrink the old Git history**.
#
# So you have two realistic student-friendly options:
#
# 1. **Accept the old Git history as-is**, and use DVC correctly from now on.
# 2. If the repo is still very early and history is not important, **rewrite history or recreate the repo cleanly** after moving data to DVC.
#
# For a student project, option 1 is usually good enough unless repo size is already causing problems.

# %% [markdown]
# ## Remote storage recommendation for a student team
#
# ### Recommended minimum choice
#
# For a student group, the most practical remote choices are:
#
# 1. **Shared Google Drive folder** — easiest conceptually for small teams
# 2. **Shared local folder / external drive / lab machine path** — easiest technically if everyone works on the same environment
# 3. A more managed remote later (for example DagsHub / object storage) if the project grows
#
# ### Option A — simplest technical setup: local/shared folder remote
#
# ```bash
# mkdir -p ../fraud-dvc-storage
# dvc remote add -d storage ../fraud-dvc-storage
# dvc push
# ```
#
# ### Option B — more collaborative: Google Drive remote
#
# ```bash
# pip install "dvc[gdrive]"
# dvc remote add -d storage gdrive://<YOUR_SHARED_FOLDER_ID>/fraud-detection-dvc
# dvc remote modify storage gdrive_acknowledge_abuse true
# dvc push
# ```
#
# ### Basic teammate workflow
#
# After cloning the repository:
#
# ```bash
# git clone <your-repo-url>
# cd Fraud-Detection
# pip install dvc
# dvc pull
# ```
#
# If using a Drive remote, teammates may need the corresponding authentication setup on their machines.

# %% [markdown]
# ## Minimal Git + DVC workflow after adoption
#
# ### First-time setup
# ```bash
# git clone <your-repo-url>
# cd Fraud-Detection
# pip install dvc
# dvc pull
# ```
#
# ### After data changes
# ```bash
# dvc add raw_data
# git add raw_data.dvc .gitignore
# git commit -m "Update raw dataset version"
# git push
# dvc push
# ```
#
# ### Later, when you create processed artifacts
# ```bash
# dvc add data/interim/train_merged.parquet
# dvc add data/processed/train_model_input.parquet
# git add data/interim/train_merged.parquet.dvc data/processed/train_model_input.parquet.dvc .gitignore
# git commit -m "Version processed fraud datasets"
# git push
# dvc push
# ```
#
# ### Once you have reproducible scripts
# At that point, add:
# - `dvc.yaml`
# - `params.yaml`
# - reproducible stages such as `prepare`, `train`, `evaluate`

# %% [markdown]
# ## Temporary data management before DVC is integrated
#
# Because the recommendation is **EDA first, DVC after**, here is the practical rule for the short gap before DVC:
#
# - keep `raw_data/` **read-only**
# - do **not** overwrite raw files
# - if you generate merged or sampled files, save them under a temporary folder such as:
#   - `data/interim/`
#   - `artifacts/tmp/`
# - add those temporary folders to `.gitignore`
# - do **not** commit large generated CSV/Parquet/model files directly into Git
#
# This avoids creating cleanup problems before DVC is introduced.

# %% [markdown]
# ## Recommended repo structure right after this step
#
# A practical minimum structure for your current repo would be:
#
# ```text
# Fraud-Detection/
# ├── raw_data/                               # keep current path for now; DVC-track it
# ├── notebooks/
# │   └── 01_fraud_eda_and_dvc_review.ipynb
# ├── src/
# │   ├── data/
# │   │   ├── make_dataset.py
# │   │   └── split_time_based.py
# │   ├── features/
# │   └── models/
# ├── data/
# │   ├── interim/                            # DVC-track when created
# │   └── processed/                          # DVC-track when created
# ├── reports/
# │   └── figures/
# ├── .dvc/
# ├── .dvcignore
# ├── dvc.yaml                                # add when stages become reproducible
# ├── params.yaml                             # add when experiments become configurable
# ├── requirements.txt
# ├── README.md
# └── .gitignore
# ```
#
# If you want **minimum disruption**, you do not need to rename `raw_data/` immediately.
# You can keep it as-is and still adopt DVC properly.

# %% [markdown]
# # Final notebook conclusion
#
# ## EDA conclusion
# This dataset should be treated as a **heavily imbalanced fraud detection problem**.  
# EDA should focus on:
#
# - class imbalance,
# - missingness patterns,
# - feature distribution differences between fraud and non-fraud,
# - outlier behavior,
# - category-specific fraud concentration,
# - and time-related drift.
#
# Those are exactly the signals that will later affect **Recall, Precision, F1-score, and PR-AUC**.
#
# ## DVC conclusion
# For the current repository state, the best decision is:
#
# > **2. Nên làm EDA trước, DVC sau**
#
# But “sau” should mean:
#
# > **Integrate DVC immediately before preprocessing / feature engineering / model artifacts start accumulating.**
#
# That gives you the best trade-off between practicality and MLOps discipline for a student project.
