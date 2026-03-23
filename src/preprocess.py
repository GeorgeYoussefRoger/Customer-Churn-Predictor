import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning for the Telco Customer Churn dataset:
    - Drop 'customerID' column
    - Convert 'TotalCharges' to numeric 
    - Map 'Churn' to binary
    - Replace 'No internet service' and 'No phone service' with 'No'
    """
    df = df.drop(columns=['customerID'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')

    return df