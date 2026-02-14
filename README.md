# Machine Learning Assignment 2 - End-to-End Classification Project

## 1. Project Structure

```text
ml_assignment_2_project/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- dummy_classification.db            # auto-created SQLite DB
|   |-- sample_test_data.csv               # auto-created during training
|-- model/
|   |-- train_models.py
|   |-- replace_dataset.py
|   |-- trainer.py
|   |-- database.py
|   |-- metrics.py
|   |-- config.py
|   |-- artifacts/                         # saved model files and metrics
```

## 2. Setup and Run

```bash
cd ml_assignment_2_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## 3. Replace Dummy DB with Your Own Dataset

The project starts with a dummy SQLite database seeded from a public dataset.

```bash
python model/replace_dataset.py --csv /path/to/your_dataset.csv --target your_target_column
```

Rules enforced for replacement dataset (as per assignment):
- at least 500 rows
- at least 12 feature columns
- classification target with at least 2 classes

## 4. Assignment README Content (Section 3 - Step 5)

### a. Problem statement
Build and compare six machine learning classification models on one dataset, evaluate each model using Accuracy, AUC, Precision, Recall, F1, and MCC, and deploy an interactive Streamlit application that supports test CSV upload, model selection, metric display, and confusion matrix/classification report.

### b. Dataset description
- Default dataset used: **Credit Card** (public UCI dataset via scikit-learn loader)
- Instances: **1579**
- Features: **18 numeric features**
- Target: **Binary class (Credit Card Approved/Rejected`)**

### c. Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

#### Comparison Table (Auto-generated after training)
Run:

```bash
python model/train_models.py
```

Generated file:
- `model/artifacts/model_comparison_metrics.csv`

Use the values from that file in your submission PDF table.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |
| Decision Tree | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |
| KNN | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |
| Naive Bayes | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |
| Random Forest (Ensemble) | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |
| XGBoost (Ensemble) | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV | Generated in CSV |

#### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline for linearly separable signals; usually stable and interpretable. |
| Decision Tree | Captures non-linear patterns but can overfit if unconstrained. |
| KNN | Performs well on local neighborhood patterns; sensitive to feature scaling and noisy data. |
| Naive Bayes | Fast and efficient baseline; may underperform when independence assumption is weak. |
| Random Forest (Ensemble) | Typically robust and high-performing due to bagging and reduced variance. |
| XGBoost (Ensemble) | Often among the best performers with boosted trees and strong handling of complex patterns. |

## 5. Streamlit Features Implemented

- Dataset upload option (`CSV`) for test data
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report
- Download predictions as CSV

## 6. Deployment on Streamlit Community Cloud

1. Push this folder to your GitHub repository.
2. Go to `https://streamlit.io/cloud`.
3. Sign in and click **New App**.
4. Select repository and branch.
5. Set main file path to: `ml_assignment_2_project/app.py`.
6. Click **Deploy**.


