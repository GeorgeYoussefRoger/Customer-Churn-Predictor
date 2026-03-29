# 📉 Customer Churn Predictor

An end-to-end Machine Learning system that predicts customer churn using a production-style pipeline, with experiment tracking, hyperparameter tuning, and a deployed API + interactive UI.

> Built on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🚀 Features

- End-to-End ML Pipeline (Preprocessing -> Training -> Tuning)
- Imbalanced Classification (PR-AUC optimized)
- Threshold Optimization for business decision-making
- Experiment Tracking with MLflow
- Hyperparameter tuning with Optuna
- FastAPI for real-time predictions
- Streamlit UI for interaction
- Dockerized services (API + UI)
- CI/CD with GitHub Actions
- Multi-service architecture (separate API & UI)

## 📦 Installation & Usage

### Prerequisites

- Python 3.12+

### Run Locally

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

4. Train Model

```
python -m src.pipeline
```

5. Run API

```
pip install -r api/requirements.txt
uvicorn api.main:app
```

6. Run UI

```
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

7. Access:
   - UI -> http://localhost:8501
   - API Docs -> http://localhost:8000/docs

### Run with Docker (Recommended)

1. Build Docker images (API + UI)

```
docker-compose up --build
```

2. Access:

- UI -> http://localhost:8501
- API Docs -> http://localhost:8000/docs

## 📊 Model Performance

- CatBoost outperformed Logistic Regression and LightGBM in PR-AUC after tuning.

- Test Set Metrics:
  - PR-AUC (Primary Metric): 0.66
  - Precision: 0.65
  - Recall: 0.52
  - F1: 0.58
  - Best Threshold: 0.41
- Notes:
  - PR-AUC was used as the primary metric due to class imbalance
  - Threshold was optimized using F1-score to balance precision and recall

## 📂 Project Structure

```
Customer-Churn-Predictor/
├── .github/workflows/     # GitHub Actions CI/CD
├── api/                   # FastAPI
├── data/                  # IBM Telco Dataset
├── models/                # Trained Models
├── notebooks/             # Exploration Notebook
├── src/                   # ML Pipeline
├── ui/                    # Streamlit UI
└── requirements.txt       # Training Requirements
```

## 📜 License

- This project is licensed under the MIT License.
- See the [LICENSE](LICENSE) file for more details.
