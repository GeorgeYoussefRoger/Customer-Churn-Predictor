DATA_DIR = 'data'
MODELS_DIR = 'models'
MLFLOW_TRACKING_URI = 'sqlite:///mlruns.db'
MLFLOW_EXPERIMENT = 'Customer-Churn-Prediction'

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']