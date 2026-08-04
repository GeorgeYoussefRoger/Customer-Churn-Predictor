import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import DATA_FILE, RANDOM_STATE, TEST_SIZE, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def preprocess():
    df = pd.read_csv(DATA_FILE)

    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    return X_train, X_test, y_train, y_test

def create_preprocessor():
    return ColumnTransformer(transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )