import optuna
import mlflow
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone

from src.config import RANDOM_STATE
from src.evaluate import evaluate

def tune(X_train, X_test, y_train, y_test, pipeline, name):
    def objective(trial):   
        if name == 'LogisticRegression':
            params = {
                'model__C': trial.suggest_float('model__C', 0.01, 100, log=True),
                'model__max_iter': trial.suggest_int('model__max_iter', 100, 1000),
                'model__class_weight': trial.suggest_categorical('model__class_weight', ['balanced', None])
            }
        elif name == 'RandomForest':
            params = {
                'model__n_estimators': trial.suggest_int('model__n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('model__max_depth', 4, 20),
                'model__min_samples_split': trial.suggest_int('model__min_samples_split', 2, 10),
                'model__min_samples_leaf': trial.suggest_int('model__min_samples_leaf', 1, 5),
            }
        elif name == 'CatBoost':
            params = {
                'model__iterations': trial.suggest_int('model__iterations', 200, 1000),
                'model__depth': trial.suggest_int('model__depth', 4, 8),
                'model__learning_rate': trial.suggest_float('model__learning_rate', 0.01, 0.2, log=True),
                'model__l2_leaf_reg': trial.suggest_float('model__l2_leaf_reg', 1, 10),
            }
        
        trial_pipeline = clone(pipeline)
        trial_pipeline.set_params(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(trial_pipeline, X_train, y_train, cv=cv, scoring='average_precision').mean()

    with mlflow.start_run(run_name=f'{name}_Tuned'):
        print(f"Tuning {name} with Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        tuned_pipeline = clone(pipeline)
        tuned_pipeline.set_params(**best_params)
        tuned_pipeline.fit(X_train, y_train)

        metrics = evaluate(tuned_pipeline, X_test, y_test)
        mlflow.log_metric("Precision", metrics["precision"])
        mlflow.log_metric("Recall", metrics["recall"])
        mlflow.log_metric("F1 Score", metrics["f1"])
        mlflow.log_metric("CV PR-AUC", study.best_value)
        mlflow.log_metric("Best Threshold", metrics["threshold"])
        mlflow.log_metric("Test PR-AUC", metrics["test_pr_auc"])

        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(tuned_pipeline, name="model")

        print(f"{name} CV PR-AUC: {study.best_value:.4f}")
        print(f"{name} Test PR-AUC: {metrics['test_pr_auc']:.4f}")

    return metrics['test_pr_auc'], tuned_pipeline, metrics['threshold']