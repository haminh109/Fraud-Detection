import argparse
import gc
import json
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    recall_score,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def evaluate_method(name, X_resampled, y_resampled, X_val, y_val, random_state=42):
    """
    Train a lightweight benchmark model on resampled data
    and evaluate on validation set.
    """
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    cls_report = classification_report(y_val, y_pred, output_dict=True)
    auprc = average_precision_score(y_val, y_proba)
    recall = recall_score(y_val, y_pred)

    logging.info("Method: %s", name)
    logging.info("Recall: %.4f", recall)
    logging.info("AUPRC: %.4f", auprc)

    return {
        "method": name,
        "recall": recall,
        "auprc": auprc,
        "classification_report": cls_report,
    }

def save_resampled_data(X_resampled, y_resampled, output_dir: Path, prefix: str):
    """
    Save balanced training data to parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_out = output_dir / f"X_train_{prefix}.parquet"
    y_out = output_dir / f"y_train_{prefix}.parquet"

    X_resampled.to_parquet(X_out, index=False)
    pd.DataFrame({"isFraud": y_resampled}).to_parquet(y_out, index=False)

    logging.info("Saved resampled X to %s", X_out)
    logging.info("Saved resampled y to %s", y_out)


def print_class_stats(y_resampled, method_name: str):
    counts = pd.Series(y_resampled).value_counts()
    percent = pd.Series(y_resampled).value_counts(normalize=True) * 100

    logging.info("--- Class distribution after %s ---", method_name)
    logging.info("Class 0 (Non-Fraud): %s samples (%.2f%%)", counts.get(0, 0), percent.get(0, 0.0))
    logging.info("Class 1 (Fraud): %s samples (%.2f%%)", counts.get(1, 0), percent.get(1, 0.0))


def parse_args():
    parser = argparse.ArgumentParser(description="Balance fraud training data and compare methods.")

    parser.add_argument(
        "--x_train_path",
        type=str,
        default="data/processed/X_train.parquet"
    )
    parser.add_argument(
        "--y_train_path",
        type=str,
        default="data/processed/y_train.parquet"
    )
    parser.add_argument(
        "--x_val_path",
        type=str,
        default="data/processed/X_val.parquet"
    )
    parser.add_argument(
        "--y_val_path",
        type=str,
        default="data/processed/y_val.parquet"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/balanced"
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="metrics/balancing_metrics.json"
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=0.3,
        help="Minority / majority ratio after resampling."
    )
    parser.add_argument(
        "--save_method",
        type=str,
        default="rus",
        choices=["rus", "smote"],
        help="Which resampled dataset to persist."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    return parser.parse_args()


def main():
    args = parse_args()

    x_train_path = Path(args.x_train_path)
    y_train_path = Path(args.y_train_path)
    x_val_path = Path(args.x_val_path)
    y_val_path = Path(args.y_val_path)
    output_dir = Path(args.output_dir)
    metrics_path = Path(args.metrics_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading processed train/validation data...")
    X_train = pd.read_parquet(x_train_path).astype("float32")
    y_train = pd.read_parquet(y_train_path).values.ravel()

    X_val = pd.read_parquet(x_val_path).astype("float32")
    y_val = pd.read_parquet(y_val_path).values.ravel()

    results = {}

    # ---------------- SMOTE ----------------
    logging.info("Running SMOTE with ratio = %.2f", args.sampling_ratio)
    smote = SMOTE(
        sampling_strategy=args.sampling_ratio,
        random_state=args.random_state
    )
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    results["smote"] = evaluate_method(
        name=f"SMOTE (ratio={args.sampling_ratio})",
        X_resampled=X_smote,
        y_resampled=y_smote,
        X_val=X_val,
        y_val=y_val,
        random_state=args.random_state
    )

    # Free memory later if needed
    gc.collect()

    # ---------------- Random Under Sampling ----------------
    logging.info("Running RandomUnderSampler with ratio = %.2f", args.sampling_ratio)
    rus = RandomUnderSampler(
        sampling_strategy=args.sampling_ratio,
        random_state=args.random_state
    )
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    results["rus"] = evaluate_method(
        name=f"RandomUnderSampler (ratio={args.sampling_ratio})",
        X_resampled=X_rus,
        y_resampled=y_rus,
        X_val=X_val,
        y_val=y_val,
        random_state=args.random_state
    )

    logging.info("=" * 40)
    logging.info("SMOTE AUPRC: %.4f", results["smote"]["auprc"])
    logging.info("RUS   AUPRC: %.4f", results["rus"]["auprc"])
    logging.info("=" * 40)

    # Print class distribution for both
    print_class_stats(y_smote, "SMOTE")
    print_class_stats(y_rus, "RUS")

    # Save chosen method
    if args.save_method == "smote":
        save_resampled_data(X_smote, y_smote, output_dir, prefix="smote")
        selected_method = "smote"
    else:
        save_resampled_data(X_rus, y_rus, output_dir, prefix="rus")
        selected_method = "rus"

    # Save metrics
    final_metrics = {
        "sampling_ratio": args.sampling_ratio,
        "selected_method": selected_method,
        "results": results,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    logging.info("Saved balancing metrics to %s", metrics_path)


if __name__ == "__main__":
    main()