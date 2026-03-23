from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = 'Churn'

NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']
                        
MODELS = {
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(n_jobs=-1, random_state=RANDOM_STATE)
}