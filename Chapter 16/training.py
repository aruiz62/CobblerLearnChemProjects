
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load housing data
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 5-fold cross-validation
cv_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

scores = cross_validate(
    cv_model,
    X_train,
    y_train,
    cv=5,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "MSE": "neg_mean_squared_error",
        "R2": "r2"
    },
    n_jobs=1
)

mae = -scores["test_MAE"]
mse = -scores["test_MSE"]
r2 = scores["test_R2"]


# Save cross-validation results
import pandas as pd
from pathlib import Path

cv_results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "MAE": mae,
    "MSE": mse,
    "R2": r2
})

cv_results.to_csv(
    Path(__file__).with_name("cv_results.csv"),
    index=False
)

print("Cross-validation results saved!")


print("5-Fold Cross-Validation Results")
print("Average MAE:", round(np.mean(mae), 4))
print("Average MSE:", round(np.mean(mse), 4))
print("Average R2:", round(np.mean(r2), 4))


# Save the trained model
joblib.dump(model, "housing_model.pkl")

print("Model trained and saved successfully!")
