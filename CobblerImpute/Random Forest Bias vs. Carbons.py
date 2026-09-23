import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=MAX_DEPTH,
    random_state=42
)

model.fit(X_train, y_train)

predicted_log = model.predict(X_test)

actual = np.expm1(y_test)
predicted = np.expm1(predicted_log)

relative_error = ((predicted - actual) / actual) * 100

plt.figure(figsize=(7, 6))

plt.scatter(X_test["carbons"], relative_error)

plt.axhline(0, linestyle="--", label="0% Error")
plt.axhline(100, linestyle=":")
plt.axhline(-100, linestyle=":")

plt.xlabel("Carbons")
plt.ylabel("Relative Error (%)")

plt.title(
    f"Random Forest (Depth = {MAX_DEPTH})\nBias vs. Carbons"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    f"random_forest_depth{MAX_DEPTH}_bias_vs_carbons.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()