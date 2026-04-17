import argparse
from pathlib import Path

import pandas as pd

from src.feature_runtime import FraudFeatureBuilder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input_path", required=True)
    parser.add_argument("--artifact_path", required=True)
    parser.add_argument("--pca_components", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_parquet(args.train_input_path)

    builder = FraudFeatureBuilder(pca_components=args.pca_components).fit(train_df)

    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    builder.save(artifact_path)

    print(f"Saved FE artifact to {artifact_path}")


if __name__ == "__main__":
    main()