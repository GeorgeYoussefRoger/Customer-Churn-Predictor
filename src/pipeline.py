import mlflow
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import *
from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train
from src.tune import tune

def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def run_pipeline():
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment('Customer-Churn-Prediction')

    df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = preprocess(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    baseline_scores = {}
    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        cv_pr_auc = train(X_train, X_test, y_train, y_test, pipeline, name)
        baseline_scores[name] = cv_pr_auc

    top2 = sorted(baseline_scores, key=baseline_scores.get, reverse=True)[:2]

    best_score = -float('inf')
    best_pipeline = None
    best_pipeline_name = None
    best_threshold = None
    for name in top2:
        model = MODELS[name]
        pipeline = build_pipeline(model)
        threshold, test_pr_auc = tune(X_train, X_test, y_train, y_test, pipeline, name)
        if test_pr_auc > best_score:
            best_score = test_pr_auc
            best_pipeline = pipeline
            best_pipeline_name = name
            best_threshold = threshold

    print(f"Best model after tuning: {best_pipeline_name} with Test PR-AUC: {best_score:.4f} at threshold {best_threshold:.4f}")
    os.makedirs('models', exist_ok=True)
    joblib.dump({"model": best_pipeline, "threshold": best_threshold}, os.path.join('models', 'final_model.pkl'))

if __name__ == "__main__":
    run_pipeline()