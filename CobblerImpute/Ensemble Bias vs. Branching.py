import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

K_NEIGHBORS = 5
MAX_DEPTH = 5

df = pd.read_csv("alkane_dataset.csv")

data = df[["carbons", "branch number", "viscosity"]].dropna()

X = data[["carbons", "branch number"]]
y = np.log1p(data["viscosity"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)
linear_prediction = linear.predict(X_test)

knn = KNeighborsRegressor(
    n_neighbors=K_NEIGHBORS
)
knn.fit(X_train, y_train)
knn_prediction = knn.predict(X_test)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=MAX_DEPTH,
    random_state=42
)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)

ensemble_log = (
    linear_prediction
    + knn_prediction
    + rf_prediction
) / 3

actual = np.expm1(y_test)
predicted = np.expm1(ensemble_log)

relative_error = ((predicted - actual) / actual) * 100

plt.figure(figsize=(7, 6))

plt.scatter(
    X_test["branch number"],
    relative_error
)

plt.axhline(0, linestyle="--", label="0% Error")
plt.axhline(100, linestyle=":")
plt.axhline(-100, linestyle=":")

plt.xlabel("Branch Number")
plt.ylabel("Relative Error (%)")

plt.title("Ensemble Model\nBias vs. Branching")

plt.legend()
plt.tight_layout()

plt.savefig(
    "ensemble_bias_vs_branching.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()