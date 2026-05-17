import os
import pandas as pd

from src.config import DATA_DIR

def load_data():
    file_path = os.path.join(DATA_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(file_path)

def clean_data(df):
    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')

    return df