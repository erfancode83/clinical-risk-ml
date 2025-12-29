from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

def build_pipeline(alpha: float = 1.0) -> Pipeline:
    """
    Build and return the ML pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=alpha))
    ])

    return pipeline
