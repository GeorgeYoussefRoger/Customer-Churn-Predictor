# 📉 Customer Churn Predictor

An end-to-end Machine Learning project that predicts customer churn using a complete workflow, from data preprocessing and model comparison to hyperparameter tuning and deployment through a FastAPI backend and Streamlit interface.

> Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🚀 Features

- End-to-End ML Pipeline (Preprocessing -> Training -> Tuning)
- Imbalanced Classification (PR-AUC optimized)
- MLflow experiment tracking
- Hyperparameter tuning with Optuna
- FastAPI inference API
- Streamlit interactive frontend
- Deployment-ready Scikit-learn pipeline

## 📦 Installation & Usage

- Prerequisites
  - Python 3.12+

1. Clone the repository

```
git clone https://github.com/GeorgeYoussefRoger/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

2. Create a Virtual Environment

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Run API

```
uvicorn api.api:app
```

5. Run UI

```
streamlit run ui/app.py
```

6. Access:
   - UI -> http://localhost:8501
   - API Docs -> http://localhost:8000/docs

## 🧠 Training

- Install dependencies

```
pip install -r requirements.txt
```

- Train Models

```
python -m src.main
```

- View MLflow experiments

```
mlflow server --backend-store-uri sqlite:///mlruns.db
```

## 🤖 Model Details

- LightGBM outperformed Logistic Regression and Random Forest in PR-AUC after tuning.

- Test Set Metrics:
  - PR-AUC (Primary Metric): 0.64
  - Precision: 0.63
  - Recall: 0.57

- Notes:
  - PR-AUC was used as the primary metric due to class imbalance
  - Three baseline models were compared using MLflow
  - The best-performing baseline model was selected for Optuna hyperparameter tuning before deployment

## 📜 License

This project is licensed under the MIT License.
