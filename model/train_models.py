from __future__ import annotations

import argparse

from model.trainer import train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all assignment models and save artifacts.")
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        help="Rebuild dummy database before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_table = train_all_models(recreate_database=args.recreate_db)

    print("Training complete. Metrics table:")
    print(comparison_table.to_string(index=False))


if __name__ == "__main__":
    main()
