import mlflow
import os
import joblib

from src.config import *
from src.preprocess import preprocess, create_preprocessor
from src.models import models, build_pipeline
from src.train import train
from src.tune import tune

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test, y_train, y_test = preprocess()
    preprocessor = create_preprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    baseline_results = {}
    for name, model in models.items():
        pr_auc = train(X_train, X_test, y_train, y_test, model, name)
        baseline_results[name] = pr_auc

    best_model_name = max(baseline_results, key=baseline_results.get)
    best_model = models[best_model_name]

    tuned_model = tune(X_train, X_test, y_train, y_test, best_model, best_model_name)
    pipeline = build_pipeline(preprocessor, tuned_model)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, os.path.join(MODELS_DIR, f"final_pipeline.pkl"))


if __name__ == "__main__":
    main()