from __future__ import annotations

import argparse
from pathlib import Path

from model.database import replace_dataset_with_csv
from model.trainer import train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the dummy SQLite dataset with your CSV and retrain all models. "
            "CSV must satisfy assignment minimums: >=500 rows and >=12 features."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to custom dataset CSV file.")
    parser.add_argument(
        "--target",
        required=True,
        help="Target column name in the provided CSV.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only replace the database; do not retrain models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()

    replace_dataset_with_csv(csv_path=csv_path, target_column=args.target)
    print(f"Database updated from: {csv_path}")

    if not args.skip_train:
        comparison_table = train_all_models(recreate_database=False)
        print("Retraining complete. Metrics table:")
        print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
