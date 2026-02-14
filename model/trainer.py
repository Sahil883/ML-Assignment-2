from __future__ import annotations

import inspect
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from model.config import (
    ARTIFACTS_DIR,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    MODEL_DISPLAY_ORDER,
    RANDOM_STATE,
    SAMPLE_TEST_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from model.database import ensure_dummy_database, load_dataset_from_db
from model.metrics import calculate_classification_metrics


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in features.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessors: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        preprocessors.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        preprocessors.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=preprocessors, remainder="drop")


def _build_model_registry(n_classes: int) -> dict[str, object]:
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for the assignment model set. "
            "Install dependencies from requirements.txt before training."
        ) from exc

    xgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    if n_classes == 2:
        xgb_params["objective"] = "binary:logistic"
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = n_classes

    models["XGBoost"] = XGBClassifier(**xgb_params)

    return models


def _slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def artifact_path_for_model(model_name: str) -> Path:
    return ARTIFACTS_DIR / f"{_slugify_model_name(model_name)}.joblib"


def all_artifacts_exist() -> bool:
    if not METRICS_CSV_PATH.exists():
        return False

    return all(artifact_path_for_model(model_name).exists() for model_name in MODEL_DISPLAY_ORDER)


def load_metrics_table() -> pd.DataFrame:
    if not METRICS_CSV_PATH.exists():
        raise FileNotFoundError(
            "Metrics file missing. Run `python model/train_models.py` from project root first."
        )

    return pd.read_csv(METRICS_CSV_PATH)


def load_model_artifact(model_name: str) -> dict:
    path = artifact_path_for_model(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact for {model_name} was not found at {path}. "
            "Run `python model/train_models.py` from project root first."
        )

    return joblib.load(path)


def _remove_rows_with_missing_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the training dataset.")

    target_series = dataset[TARGET_COLUMN]
    target_as_string = target_series.astype("string").str.strip()
    missing_target_mask = target_series.isna() | target_as_string.eq("").fillna(False)

    cleaned_dataset = dataset.loc[~missing_target_mask].copy()
    dropped_rows = int(missing_target_mask.sum())

    if cleaned_dataset.empty:
        raise ValueError("Training dataset has no valid target labels after removing missing labels.")

    class_count = cleaned_dataset[TARGET_COLUMN].nunique(dropna=True)
    if class_count < 2:
        raise ValueError(
            "Training dataset must contain at least two target classes after removing missing labels."
        )

    return cleaned_dataset, dropped_rows


def _oversample_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    training_frame = x_train.copy()
    training_frame[TARGET_COLUMN] = y_train

    class_counts = training_frame[TARGET_COLUMN].value_counts(dropna=False)
    target_class_size = int(class_counts.max())
    balanced_slices: list[pd.DataFrame] = []

    for class_value in class_counts.index.tolist():
        class_slice = training_frame[training_frame[TARGET_COLUMN] == class_value]
        balanced_slices.append(
            class_slice.sample(
                n=target_class_size,
                replace=len(class_slice) < target_class_size,
                random_state=RANDOM_STATE,
            )
        )

    balanced_frame = pd.concat(balanced_slices, ignore_index=True)
    balanced_frame = balanced_frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    balanced_target = balanced_frame[TARGET_COLUMN].copy()
    balanced_features = balanced_frame.drop(columns=[TARGET_COLUMN])

    return balanced_features, balanced_target


def _fit_pipeline_with_imbalance_handling(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> str:
    model = pipeline.named_steps["model"]

    supports_sample_weight = False
    try:
        supports_sample_weight = "sample_weight" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        supports_sample_weight = False

    if supports_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)
        return "sample_weight_balancing"

    balanced_x_train, balanced_y_train = _oversample_training_data(x_train=x_train, y_train=y_train)
    pipeline.fit(balanced_x_train, balanced_y_train)
    return "random_oversampling"


def train_all_models(recreate_database: bool = False) -> pd.DataFrame:
    ensure_dummy_database(recreate=recreate_database)

    raw_dataset = load_dataset_from_db()
    dataset, dropped_missing_target_rows = _remove_rows_with_missing_target(raw_dataset)
    features = dataset.drop(columns=[TARGET_COLUMN])
    target = dataset[TARGET_COLUMN].astype(str)

    label_encoder = LabelEncoder()
    encoded_target = pd.Series(label_encoder.fit_transform(target), index=target.index, name=TARGET_COLUMN)
    class_labels = label_encoder.classes_.tolist()
    n_classes = len(class_labels)
    class_distribution = {str(class_label): int((target == class_label).sum()) for class_label in class_labels}

    min_class_count = int(encoded_target.value_counts(dropna=False).min())
    stratify_target = encoded_target if min_class_count >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    test_export = x_test.copy()
    test_export[TARGET_COLUMN] = label_encoder.inverse_transform(y_test)
    test_export.to_csv(SAMPLE_TEST_DATA_PATH, index=False)

    comparison_rows: list[dict[str, float | str]] = []

    for model_name, estimator in _build_model_registry(n_classes=n_classes).items():
        preprocessor = _build_preprocessor(features)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        imbalance_strategy = _fit_pipeline_with_imbalance_handling(
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
        )
        predictions = pipeline.predict(x_test)

        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(x_test)

        metrics = calculate_classification_metrics(
            y_true=y_test,
            y_pred=predictions,
            y_probabilities=probabilities,
            n_classes=n_classes,
        )

        report = classification_report(
            y_test,
            predictions,
            labels=list(range(n_classes)),
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=list(range(n_classes)))

        artifact_payload = {
            "model_name": model_name,
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "feature_columns": list(features.columns),
            "target_column": TARGET_COLUMN,
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": matrix,
            "class_labels": class_labels,
            "dataset_shape": dataset.shape,
            "dropped_missing_target_rows": dropped_missing_target_rows,
            "class_distribution": class_distribution,
            "imbalance_strategy": imbalance_strategy,
        }

        joblib.dump(artifact_payload, artifact_path_for_model(model_name))

        comparison_rows.append(
            {
                "ML Model Name": model_name,
                "Accuracy": metrics["Accuracy"],
                "AUC": metrics["AUC"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "MCC": metrics["MCC"],
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table["ML Model Name"] = pd.Categorical(
        comparison_table["ML Model Name"],
        categories=MODEL_DISPLAY_ORDER,
        ordered=True,
    )
    comparison_table = comparison_table.sort_values("ML Model Name").reset_index(drop=True)

    comparison_table.to_csv(METRICS_CSV_PATH, index=False)
    comparison_table.to_json(METRICS_JSON_PATH, orient="records", indent=2)

    return comparison_table
