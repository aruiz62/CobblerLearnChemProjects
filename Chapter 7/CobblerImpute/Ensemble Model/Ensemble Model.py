import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =====================================
# SETTINGS YOU CAN CHANGE
# =====================================

K_NEIGHBORS = 5
MAX_DEPTH = 5


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("../alkane_dataset.csv")

data = df[
    ["carbons", "branch number", "viscosity"]
].dropna()


# =====================================
# FEATURES AND TARGET
# =====================================

X = data[
    ["carbons", "branch number"]
]

# Log-transform viscosity
y = np.log1p(data["viscosity"])


# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# MODEL 1: LINEAR REGRESSION
# =====================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(X_test)


# =====================================
# MODEL 2: KNN
# =====================================

knn_model = KNeighborsRegressor(
    n_neighbors=K_NEIGHBORS
)

knn_model.fit(
    X_train,
    y_train
)

knn_predictions = knn_model.predict(X_test)


# =====================================
# MODEL 3: RANDOM FOREST
# =====================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=MAX_DEPTH,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(X_test)


# =====================================
# ENSEMBLE AVERAGE
# =====================================

ensemble_predictions = (
    linear_predictions
    + knn_predictions
    + rf_predictions
) / 3


# =====================================
# CALCULATE MAE
# =====================================

mae = mean_absolute_error(
    y_test,
    ensemble_predictions
)

print("Ensemble Average Model")
print(f"KNN Neighbors: {K_NEIGHBORS}")
print(f"Random Forest Depth: {MAX_DEPTH}")
print(f"MAE: {mae:.4f}")


# =====================================
# PREDICTION QUALITY GRAPH
# =====================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    ensemble_predictions
)


# Perfect prediction line
minimum = min(
    y_test.min(),
    ensemble_predictions.min()
)

maximum = max(
    y_test.max(),
    ensemble_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)


# =====================================
# GRAPH LABELS
# =====================================

plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title(
    "Ensemble Model Prediction Quality"
)


# Put MAE on graph
plt.text(
    0.05,
    0.92,
    f"MAE = {mae:.4f}",
    transform=plt.gca().transAxes,
    fontsize=12
)

plt.legend()
plt.tight_layout()


# =====================================
# SAVE GRAPH
# =====================================

plt.savefig(
    "ensemble_model.png",
    dpi=300,
    bbox_inches="tight"
)


# Show graph
plt.show()