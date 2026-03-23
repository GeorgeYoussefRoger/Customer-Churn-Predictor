import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(file_path)