import pandas as pd

def extract_coefficients(pipeline, feature_names) -> pd.DataFrame:
    """
    Extract and sort model coefficients by absolute impact.
    """
    ridge = pipeline.named_steps["ridge"]
    poly = pipeline.named_steps["poly"]

    expanded_features = poly.get_feature_names_out(feature_names)
    coefficients = ridge.coef_

    coef_df = pd.DataFrame({
        "feature": expanded_features,
        "coefficient": coefficients
    })

    return coef_df.sort_values(
        by="coefficient",
        key=abs,
        ascending=False
    )
