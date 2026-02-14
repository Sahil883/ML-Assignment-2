from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from model.config import DB_PATH, TABLE_NAME, TARGET_COLUMN


class DatasetValidationError(ValueError):
    """Raised when a replacement dataset does not meet assignment constraints."""


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dummy_database(recreate: bool = False) -> None:
    """Seed SQLite DB with a public classification dataset if DB does not exist."""
    if DB_PATH.exists() and not recreate:
        return

    _ensure_parent_dir(DB_PATH)

    features, target = load_breast_cancer(return_X_y=True, as_frame=True)
    dataset = features.copy()
    dataset[TARGET_COLUMN] = target.astype(int)

    with sqlite3.connect(DB_PATH) as connection:
        dataset.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)


def load_dataset_from_db() -> pd.DataFrame:
    """Load the active training dataset from SQLite."""
    ensure_dummy_database(recreate=False)

    with sqlite3.connect(DB_PATH) as connection:
        query = f"SELECT * FROM {TABLE_NAME}"
        dataset = pd.read_sql_query(query, connection)

    return dataset


def validate_dataset_for_assignment(dataset: pd.DataFrame, target_column: str) -> None:
    """Validate minimum assignment constraints for custom training datasets."""
    if target_column not in dataset.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' not found in the provided CSV."
        )

    feature_count = dataset.drop(columns=[target_column]).shape[1]
    instance_count = dataset.shape[0]
    class_count = dataset[target_column].nunique(dropna=True)

    if feature_count < 12:
        raise DatasetValidationError(
            f"Dataset must have at least 12 features; found {feature_count}."
        )

    if instance_count < 500:
        raise DatasetValidationError(
            f"Dataset must have at least 500 rows; found {instance_count}."
        )

    if class_count < 2:
        raise DatasetValidationError("Classification target must contain at least two classes.")


def replace_dataset_with_csv(csv_path: Path, target_column: str) -> pd.DataFrame:
    """Replace DB table with user CSV and normalize target column name."""
    dataset = pd.read_csv(csv_path)
    validate_dataset_for_assignment(dataset=dataset, target_column=target_column)

    normalized_dataset = dataset.rename(columns={target_column: TARGET_COLUMN}).copy()

    with sqlite3.connect(DB_PATH) as connection:
        normalized_dataset.to_sql(TABLE_NAME, connection, if_exists="replace", index=False)

    return normalized_dataset
