import mlflow
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

from src.config import ( 
    RANDOM_STATE, TEST_SIZE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODELS_DIR, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from src.preprocess import load_data
from src.preprocess import clean_data
from src.train import train
from src.tune import tune

MODELS = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
}

def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ])

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    baseline_scores = {}
    for name, model in MODELS.items():
        pipeline = build_pipeline(model)
        cv_pr_auc = train(X_train, X_test, y_train, y_test, pipeline, name)
        baseline_scores[name] = cv_pr_auc

    print("Baseline Scores:")
    for name, score in baseline_scores.items():
        print(f"  {name}: {score:.4f}")
    top2 = sorted(baseline_scores, key=baseline_scores.get, reverse=True)[:2]

    best_score = -float('inf')
    best_pipeline = None
    best_pipeline_name = None
    best_threshold = None
    for name in top2:
        model = MODELS[name]
        pipeline = build_pipeline(model)
        test_pr_auc, tuned_pipeline, threshold = tune(X_train, X_test, y_train, y_test, pipeline, name)
        if test_pr_auc > best_score:
            best_score = test_pr_auc
            best_pipeline = tuned_pipeline
            best_pipeline_name = name
            best_threshold = threshold

    print(f"Best model: {best_pipeline_name} with Test PR-AUC: {best_score:.4f} at threshold {best_threshold:.4f}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": best_pipeline, "threshold": best_threshold}, os.path.join(MODELS_DIR, 'final_model.pkl'))

if __name__ == "__main__":
    main()