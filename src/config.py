from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']
                        
MODELS = {
    'LogisticRegression': LogisticRegression(),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=RANDOM_STATE),
    'LightGBM': LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE)
}