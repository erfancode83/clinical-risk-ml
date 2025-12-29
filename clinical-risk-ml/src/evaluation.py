from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np

def cross_validated_mae(model, X, y, cv: int = 5) -> np.ndarray:
    """
    Perform cross-validation and return MAE scores.
    """
    scores = cross_val_score(
        model,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=cv
    )

    return -scores


def group_mae(df):
    """
    Calculate MAE per risk group.
    """
    return df.groupby("risk_group").apply(
        lambda g: mean_absolute_error(
            g["risk_score_true"],
            g["risk_score_pred"]
        )
    )
