# ======================================
# 1. Imports
# ======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay


# ======================================
# 2. Clinical Dataset
# ======================================
data = {
    'age': [25, 35, 45, 50, 23, 60, 33, 47, 55, 29, 40, 65, 38, 52, 41, 30, 58, 26, 49, 37],
    'bmi': [22.1, 26.5, 29.8, 31.0, 20.5, 33.2, 25.0, 28.7, 30.5, 24.3,
            27.6, 34.1, 26.9, 32.0, 27.8, 23.8, 31.5, 21.9, 29.0, 25.7],
    'blood_pressure': [110, 125, 140, 145, 105, 160, 120, 135, 150, 115,
                       130, 165, 128, 142, 132, 118, 155, 112, 138, 124],
    'glucose': [85, 95, 110, 120, 80, 140, 90, 105, 125, 88,
                100, 135, 98, 115, 102, 87, 130, 82, 112, 96],
    'risk_score': [1.2, 2.5, 4.1, 5.0, 0.8, 6.8, 2.0, 3.7, 5.5, 1.5,
                   3.2, 7.2, 2.8, 5.1, 3.5, 1.8, 6.0, 1.0, 4.3, 2.6]
}

df = pd.DataFrame(data)

X = df.drop(columns='risk_score')
y = df['risk_score']


# ======================================
# 3. Medical ML Pipeline
# ======================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])


# ======================================
# 4. Cross-Validation (MAE)
# ======================================
cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    scoring='neg_mean_absolute_error',
    cv=5
)

mae_scores = -cv_scores

print("Cross-Validation MAE per fold:")
for i, mae in enumerate(mae_scores, 1):
    print(f"  Fold {i}: {mae:.4f}")

print(f"\nMean MAE: {mae_scores.mean():.4f}")
print(f"STD MAE: {mae_scores.std():.4f}")


# ======================================
# 5. Train Final Model
# ======================================
pipeline.fit(X, y)
y_pred = pipeline.predict(X)


# ======================================
# 6. Error Analysis by Risk Group
# ======================================
analysis_df = pd.DataFrame({
    'risk_score_true': y,
    'risk_score_pred': y_pred
})

low_q = analysis_df['risk_score_true'].quantile(0.33)
high_q = analysis_df['risk_score_true'].quantile(0.66)

def risk_group(val):
    if val <= low_q:
        return 'Low Risk'
    elif val <= high_q:
        return 'Medium Risk'
    else:
        return 'High Risk'

analysis_df['risk_group'] = analysis_df['risk_score_true'].apply(risk_group)

group_mae = (
    analysis_df
    .groupby('risk_group')
    .apply(lambda df: mean_absolute_error(df['risk_score_true'], df['risk_score_pred']))
)

print("\nMAE by Risk Group:")
print(group_mae)


# ======================================
# 7. Explainability — Coefficients
# ======================================
ridge = pipeline.named_steps['ridge']
poly = pipeline.named_steps['poly']

feature_names = poly.get_feature_names_out(X.columns)
coefficients = ridge.coef_

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values(by='coefficient', key=abs, ascending=False)

print("\nTop Model Coefficients:")
print(coef_df.head(10))


# ======================================
# 8. Visualization — Main Clinical Features
# ======================================
main_features = ['age', 'bmi', 'blood_pressure', 'glucose']
main_coef = coef_df[coef_df['feature'].isin(main_features)]

plt.figure(figsize=(8, 5))
plt.barh(main_coef['feature'], main_coef['coefficient'])
plt.xlabel("Coefficient Value")
plt.title("Impact of Main Clinical Features on Risk Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ======================================
# 9. Partial Dependence Plot (BMI)
# ======================================
PartialDependenceDisplay.from_estimator(
    pipeline,
    X,
    features=['bmi'],
    grid_resolution=50
)

plt.title("Partial Dependence of BMI on Risk Score")
plt.show()
