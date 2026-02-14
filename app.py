from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.config import METRICS_CSV_PATH, MODEL_DISPLAY_ORDER, TARGET_COLUMN
from model.database import ensure_dummy_database, load_dataset_from_db
from model.metrics import calculate_classification_metrics
from model.trainer import all_artifacts_exist, load_metrics_table, load_model_artifact, train_all_models


st.set_page_config(page_title="ML Assignment 2 - Classification Dashboard", layout="wide")


def ensure_project_is_ready() -> pd.DataFrame:
    ensure_dummy_database(recreate=False)

    if not all_artifacts_exist() or not METRICS_CSV_PATH.exists():
        with st.spinner("Training models and generating artifacts..."):
            train_all_models(recreate_database=False)

    return load_metrics_table()


def render_confusion_matrix(matrix: pd.DataFrame, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    st.pyplot(figure)
    plt.close(figure)


def build_report_dataframe(report_dict: dict) -> pd.DataFrame:
    report_df = pd.DataFrame(report_dict).transpose()
    numeric_columns = [
        column
        for column in ["precision", "recall", "f1-score", "support"]
        if column in report_df.columns
    ]
    if numeric_columns:
        report_df[numeric_columns] = report_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return report_df


def _normalize_column_name(column_name: object) -> str:
    return str(column_name).strip().lower()


def _resolve_feature_columns(
    uploaded_df: pd.DataFrame,
    required_features: list[str],
) -> tuple[dict[str, str], list[str]]:
    normalized_to_uploaded: dict[str, str] = {}
    for column in uploaded_df.columns:
        normalized_to_uploaded.setdefault(_normalize_column_name(column), column)

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for feature in required_features:
        matched_column = normalized_to_uploaded.get(_normalize_column_name(feature))
        if matched_column is None:
            missing.append(feature)
        else:
            mapping[feature] = matched_column

    return mapping, missing


def _resolve_target_column(
    uploaded_df: pd.DataFrame,
    expected_target_column: str,
    used_feature_columns: list[str],
) -> str | None:
    normalized_to_uploaded: dict[str, str] = {}
    for column in uploaded_df.columns:
        normalized_to_uploaded.setdefault(_normalize_column_name(column), column)

    target_candidates = [
        expected_target_column,
        "label",
        "labels",
        "class",
        "class_label",
        "y",
        "output",
    ]

    for candidate in target_candidates:
        matched_column = normalized_to_uploaded.get(_normalize_column_name(candidate))
        if matched_column is not None:
            return matched_column

    non_feature_columns = [column for column in uploaded_df.columns if column not in used_feature_columns]
    if len(non_feature_columns) == 1:
        return non_feature_columns[0]

    return None


def show_dataset_summary() -> None:
    dataset = load_dataset_from_db()
    st.subheader("Training Dataset Summary")

    left, right = st.columns(2)
    with left:
        st.write(f"Rows: {dataset.shape[0]}")
        st.write(f"Columns: {dataset.shape[1]}")
        st.write(f"Feature count: {dataset.shape[1] - 1}")
    with right:
        target_distribution = dataset[TARGET_COLUMN].value_counts(dropna=False)
        st.write("Target distribution:")
        st.dataframe(target_distribution.rename("count"))

    st.caption("Active source: SQLite dummy database. Replace with your own CSV using model/replace_dataset.py.")


def show_model_details(model_name: str, artifact: dict) -> None:
    st.subheader(f"Selected Model: {model_name}")

    metric_columns = st.columns(6)
    for column, metric_name in zip(
        metric_columns,
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        strict=True,
    ):
        metric_value = artifact["metrics"].get(metric_name)
        column.metric(metric_name, f"{metric_value:.4f}" if pd.notna(metric_value) else "NaN")

    class_labels = artifact["class_labels"]
    confusion_matrix_df = pd.DataFrame(
        artifact["confusion_matrix"],
        index=class_labels,
        columns=class_labels,
    )

    report_df = build_report_dataframe(artifact["classification_report"])

    left, right = st.columns(2)
    with left:
        render_confusion_matrix(confusion_matrix_df, f"{model_name} - Confusion Matrix")
    with right:
        st.write("Classification Report")
        st.dataframe(report_df, use_container_width=True)


def evaluate_uploaded_test_data(model_name: str, artifact: dict) -> None:
    st.subheader("Dataset Upload (CSV)")
    st.caption(
        "Upload test data only. Include the `target` column if you want evaluation metrics; "
        "otherwise only predictions will be generated."
    )

    uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])
    if uploaded_file is None:
        return

    uploaded_df = pd.read_csv(uploaded_file)
    required_features = artifact["feature_columns"]
    feature_column_mapping, missing_features = _resolve_feature_columns(
        uploaded_df=uploaded_df,
        required_features=required_features,
    )

    if missing_features:
        st.error(
            "Missing required feature columns: "
            + ", ".join(missing_features[:15])
            + (" ..." if len(missing_features) > 15 else "")
        )
        return

    source_feature_columns = [feature_column_mapping[column] for column in required_features]
    features = uploaded_df[source_feature_columns].copy()
    features.columns = required_features
    model_pipeline = artifact["pipeline"]
    label_encoder = artifact["label_encoder"]
    class_labels = artifact["class_labels"]

    predicted_encoded = model_pipeline.predict(features)
    predicted_labels = label_encoder.inverse_transform(predicted_encoded)

    predictions_df = uploaded_df.copy()
    predictions_df["predicted_target"] = predicted_labels

    st.write("Preview with Predictions")
    st.dataframe(predictions_df.head(20), use_container_width=True)

    csv_buffer = io.StringIO()
    predictions_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download predictions CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{model_name.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

    expected_target_column = str(artifact.get("target_column", TARGET_COLUMN))
    target_column = _resolve_target_column(
        uploaded_df=uploaded_df,
        expected_target_column=expected_target_column,
        used_feature_columns=source_feature_columns,
    )
    if target_column is None:
        st.info(
            "No target column detected in upload, so evaluation metrics were skipped. "
            "Use `target` (or aliases like `label`, `class`, `y`) in the CSV."
        )
        return

    if _normalize_column_name(target_column) != _normalize_column_name(expected_target_column):
        st.info(
            f"Detected `{target_column}` as target column for evaluation "
            f"(expected `{expected_target_column}`)."
        )

    target_series = uploaded_df[target_column]
    target_as_string = target_series.astype("string").str.strip()
    valid_target_mask = target_series.notna() & target_as_string.ne("").fillna(False)
    skipped_target_rows = int((~valid_target_mask).sum())

    if skipped_target_rows:
        st.warning(
            f"Skipped {skipped_target_rows} row(s) with missing or blank target labels "
            "while computing uploaded-data metrics."
        )

    if valid_target_mask.sum() == 0:
        st.info("No valid target labels were found in uploaded data, so evaluation metrics were skipped.")
        return

    true_labels = target_as_string[valid_target_mask].astype(str)
    predicted_encoded_for_eval = predicted_encoded[valid_target_mask.to_numpy()]
    features_for_eval = features.loc[valid_target_mask]

    try:
        true_encoded = label_encoder.transform(true_labels)
    except ValueError as error:
        st.error(
            "Uploaded target column contains labels not seen during training. "
            f"Details: {error}"
        )
        return

    probabilities = None
    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(features_for_eval)

    upload_metrics = calculate_classification_metrics(
        y_true=true_encoded,
        y_pred=predicted_encoded_for_eval,
        y_probabilities=probabilities,
        n_classes=len(class_labels),
    )

    st.write("Evaluation Metrics on Uploaded Data")
    metrics_frame = pd.DataFrame([upload_metrics])
    st.dataframe(metrics_frame, use_container_width=True)

    upload_confusion_matrix = confusion_matrix(
        true_encoded,
        predicted_encoded_for_eval,
        labels=list(range(len(class_labels))),
    )
    upload_confusion_matrix_df = pd.DataFrame(
        upload_confusion_matrix,
        index=class_labels,
        columns=class_labels,
    )
    render_confusion_matrix(upload_confusion_matrix_df, f"{model_name} - Uploaded Data Confusion Matrix")

    report = classification_report(
        true_encoded,
        predicted_encoded_for_eval,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    st.write("Classification Report on Uploaded Data")
    st.dataframe(build_report_dataframe(report), use_container_width=True)


def main() -> None:
    st.title("Machine Learning Assignment 2 - End-to-End Classification Project")

    st.write(
        "This app trains and compares six classification models, stores artifacts, and allows "
        "test-data upload for inference and evaluation."
    )

    try:
        metrics_table = ensure_project_is_ready()
    except ImportError as error:
        st.error(f"Dependency error: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Project initialization failed: {error}")
        st.stop()

    show_dataset_summary()

    st.subheader("Model Comparison Table")
    st.dataframe(metrics_table, use_container_width=True)

    selected_model = st.selectbox("Model selection dropdown", MODEL_DISPLAY_ORDER)
    selected_artifact = load_model_artifact(selected_model)

    show_model_details(selected_model, selected_artifact)
    evaluate_uploaded_test_data(selected_model, selected_artifact)


if __name__ == "__main__":
    main()
