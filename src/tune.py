import optuna
import mlflow
from sklearn.model_selection import cross_val_score

from src.config import RANDOM_STATE
from src.evaluate import evaluate

def tune(X_train, X_test, y_train, y_test, pipeline, name):
    """
    Tune best model using Optuna and log results to MLflow.
    """
    def objective(trial):   
        if name == 'LogisticRegression':
            params = {
                'model__C': trial.suggest_float('model__C', 0.01, 100, log=True),
                'model__penalty': trial.suggest_categorical('model__penalty', ['l1', 'l2']),
                'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
                'model__max_iter': 1000,
                'model__solver': 'liblinear',
                'model__random_state': RANDOM_STATE
            }
        elif name == 'RandomForest':
            params = {
                'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 500),
                'model__max_depth': trial.suggest_int('model__max_depth', 5, 30),
                'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 20),
                'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 10),
                'model__max_features': trial.suggest_categorical('model__max_features', ['sqrt', 'log2', None]),
                'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None]),
                'model__random_state': RANDOM_STATE,
                'model__n_jobs': -1
            }
        elif name == 'XGBoost':
            params = {
                'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('model__max_depth', 3, 10),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.3, log=True),
                'model__subsample': trial.suggest_float('model__subsample', 0.5, 1.0),
                'model__colsample_bytree': trial.suggest_float('model__colsample_bytree', 0.5, 1.0),
                'model__scale_pos_weight': trial.suggest_float('model__scale_pos_weight', 1, 3),
                'model__eval_metric': 'logloss',
                'model__random_state': RANDOM_STATE,
                'model__n_jobs': -1
            }
        pipeline.set_params(**params)
        return cross_val_score(pipeline, X_train, y_train, cv=5, scoring='average_precision').mean()

    with mlflow.start_run(run_name=f'{name}_Tuned'):
        print(f"Tuning {name} with Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)

        best_params = study.best_params
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        mlflow.log_params(best_params)
        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metric("Precision", metrics["precision"])
        mlflow.log_metric("Recall", metrics["recall"])
        mlflow.log_metric("F1 Score", metrics["f1"])
        mlflow.log_metric("CV PR-AUC", study.best_value)
        mlflow.log_metric("Best Threshold", metrics["threshold"])
        mlflow.log_metric("Test PR-AUC", metrics["test_pr_auc"])

        mlflow.log_param("model", name)
        mlflow.sklearn.log_model(pipeline, name="model")
        print(f"{name} CV PR-AUC: {study.best_value:.4f}")
        print(f"{name} Test PR-AUC: {metrics['test_pr_auc']:.4f}")

    return metrics['threshold'], metrics['test_pr_auc']