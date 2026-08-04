from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE

models = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    'LightGBM': LGBMClassifier(verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE)
}

def build_pipeline(preprocessor, model):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline