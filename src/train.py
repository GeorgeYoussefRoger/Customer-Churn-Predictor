import mlflow

from src.evaluate import evaluate

def train(X_train, X_test, y_train, y_test, model, name):
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_metrics({
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Test PR-AUC": metrics["PR-AUC"]
        })

        print(f"{name} Test PR-AUC: {metrics['PR-AUC']:.4f}")
        return metrics["PR-AUC"]