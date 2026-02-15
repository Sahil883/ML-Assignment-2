# Machine Learning Assignment 2 - End-to-End Classification Project

## a. Problem statement
Build and compare six machine learning classification models on one dataset, evaluate each model using Accuracy, AUC, Precision, Recall, F1, and MCC, and deploy an interactive Streamlit application for prediction and evaluation.

## b. Dataset description
- Dataset used: custom classification dataset loaded from CSV into SQLite (`data/dummy_classification.db`) using `model/replace_dataset.py`.
- Total rows in source dataset: `3096`.
- Total input features: `18` (plus 1 target column).
- Target column used by the pipeline: `target` (binary classes: `0.0`, `1.0`).
- Missing/blank target labels: `1548` rows were removed before model training.
- Final rows used for model training: `1548`.
- Class distribution after cleaning: class `0.0` = `1373`, class `1.0` = `175` (imbalanced dataset).
- Preprocessing in pipeline:
- Numeric features: median imputation + standard scaling.
- Categorical features: most-frequent imputation + one-hot encoding.
- Imbalance handling in training:
- Models with `sample_weight` support use class-balanced sample weights.
- KNN uses random oversampling fallback.

## c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (evaluation metrics for all 6 models)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5452 | 0.5089 | 0.8069 | 0.5452 | 0.628 | 0.0244 |
| Decision Tree | 0.8935 | 0.7281 | 0.8922 | 0.8935 | 0.8929 | 0.4619 |
| KNN | 0.7581 | 0.7963 | 0.8781 | 0.7581 | 0.7978 | 0.3332 |
| Naive Bayes | 0.6581 | 0.5916 | 0.8232 | 0.6581 | 0.7182 | 0.0946 |
| Random Forest | 0.8935 | 0.7995 | 0.8922 | 0.8935 | 0.8929 | 0.4619 |
| XGBoost | 0.771 | 0.7103 | 0.852 | 0.771 | 0.8023 | 0.2383 |

Metrics source: `model/artifacts/model_comparison_metrics.csv`.

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Lowest overall performance (Accuracy 0.5452, MCC 0.0244), indicating underfitting for this dataset. |
| Decision Tree | Strong overall performance with high Accuracy/F1 and high MCC (0.4619). |
| KNN | Better than linear baseline; strongest AUC among non-ensemble tree methods in this run (0.7963). |
| Naive Bayes | Moderate performance; better than Logistic Regression but below tree/ensemble methods. |
| Random Forest (Ensemble) | Best AUC (0.7995) and tied-best Accuracy/F1 with Decision Tree; most reliable overall model here. |
| XGBoost (Ensemble) | Good but below Random Forest/Decision Tree in this run; still outperforms simpler baselines on most metrics. |

## Project structure

```text
ml_assignment_2_project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   |-- dummy_classification.db
|   |-- sample_test_data.csv
|-- model/
|   |-- train_models.py
|   |-- replace_dataset.py
|   |-- trainer.py
|   |-- database.py
|   |-- metrics.py
|   |-- config.py
|   |-- artifacts/
|       |-- *.joblib
|       |-- model_comparison_metrics.csv
|       |-- model_comparison_metrics.json
```

## Setup and run

```bash
cd ml_assignment_2_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m model.train_models
streamlit run app.py
```

## Replace dataset and retrain

```bash
python -m model.replace_dataset --csv /path/to/your_dataset.csv --target your_target_column
```

Validation rules (as required in assignment):
- at least 500 rows
- at least 12 feature columns
- at least 2 target classes

## Streamlit features implemented
- Dataset upload option (`CSV`) for test data.
- Model selection dropdown.
- Display of evaluation metrics.
- Confusion matrix and classification report.
- Download predictions as CSV.

## Deployment steps (Streamlit Community Cloud)
1. Go to `https://streamlit.io/cloud`.
2. Sign in with GitHub.
3. Click `New App`.
4. Select repository and branch.
5. Set main file path to `app.py`.
6. Click `Deploy`.

## Submission checklist
1. GitHub repository link added in submitted PDF.
2. Live Streamlit app link added in submitted PDF.
3. One screenshot from BITS Virtual Lab execution added in submitted PDF.
4. This README content copied into the submitted PDF (as required).
