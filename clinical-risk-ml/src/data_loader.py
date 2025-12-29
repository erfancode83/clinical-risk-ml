import pandas as pd
from pathlib import Path

def load_clinical_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load clinical dataset and split into features (X) and target (y).
    """
    df = pd.read_csv(csv_path)

    X = df.drop(columns="risk_score")
    y = df["risk_score"]

    return X, y
