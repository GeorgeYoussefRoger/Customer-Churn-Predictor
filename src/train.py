import mlflow
from sklearn.model_selection import cross_val_score

from src.evaluate import evaluate

def train(X_train, X_test, y_train, y_test, pipeline, name):
    """
    Train models and log results to MLflow.
    """
    with mlflow.start_run(run_name=f'{name}_Baseline'):

        print(f"Training {name}...")
        cv_pr_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='average_precision').mean()
        pipeline.fit(X_train, y_train)
            
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("Precision", metrics["precision"])
        mlflow.log_metric("Recall", metrics["recall"])
        mlflow.log_metric("F1 Score", metrics["f1"])
        mlflow.log_metric("CV PR-AUC", cv_pr_auc)

        mlflow.log_param("model", name)
        mlflow.sklearn.log_model(pipeline, name="model")
        print(f"{name} CV PR-AUC: {cv_pr_auc:.4f}")

    return cv_pr_auc