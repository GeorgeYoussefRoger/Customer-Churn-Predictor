import optuna
import mlflow
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import RANDOM_STATE
from src.evaluate import evaluate

def tune(X_train, X_test, y_train, y_test, model, name):
    def objective(trial):
        if name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
            }
        elif name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 4, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
        elif name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            }

        model.set_params(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision').mean()

    with mlflow.start_run(run_name=f"{name}_tuned"):
        print(f"Tuning {name}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        best_params = study.best_params
        tuned_model = model.set_params(**best_params)
        tuned_model.fit(X_train, y_train)

        metrics = evaluate(tuned_model, X_test, y_test)
            
        mlflow.log_metrics({
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "CV PR-AUC": study.best_value,
            "Test PR-AUC": metrics["PR-AUC"]
        })

        mlflow.log_params(best_params)
        print(f"Best parameters for {name}: {best_params}")
        print(f"{name} CV PR-AUC: {study.best_value:.4f}")
        print(f"{name} Test PR-AUC: {metrics['PR-AUC']:.4f}")
        
        return tuned_model