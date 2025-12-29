import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_main_coefficients(coef_df, features):
    """
    Plot coefficients of main clinical features.
    """
    subset = coef_df[coef_df["feature"].isin(features)]

    plt.figure(figsize=(8, 5))
    plt.barh(subset["feature"], subset["coefficient"])
    plt.xlabel("Coefficient Value")
    plt.title("Impact of Main Clinical Features")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_partial_dependence(model, X, feature):
    """
    Plot partial dependence for a given feature.
    """
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        features=[feature],
        grid_resolution=50
    )
    plt.show()
