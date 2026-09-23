import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =====================================
# CHANGE RANDOM FOREST DEPTH HERE
# =====================================
MAX_DEPTH = 15

# Change 5 to 15, 20, etc. whenever needed


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
# SPLIT TRAINING AND TESTING DATA
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# CREATE RANDOM FOREST MODEL
# =====================================
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=MAX_DEPTH,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


# =====================================
# MAKE PREDICTIONS
# =====================================
predicted = model.predict(X_test)


# =====================================
# CALCULATE MAE
# =====================================
mae = mean_absolute_error(
    y_test,
    predicted
)

print(f"Maximum Tree Depth: {MAX_DEPTH}")
print(f"MAE: {mae:.4f}")


# =====================================
# CREATE PREDICTION QUALITY GRAPH
# =====================================
plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    predicted
)


# Perfect prediction line
minimum = min(
    y_test.min(),
    predicted.min()
)

maximum = max(
    y_test.max(),
    predicted.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--",
    label="Perfect Prediction"
)


# Graph labels
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.title(
    f"Random Forest Prediction Quality\n"
    f"Maximum Tree Depth = {MAX_DEPTH}"
)


# Put MAE directly on graph
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
# SAVE GRAPH AUTOMATICALLY
# =====================================
plt.savefig(
    f"random_forest_depth{MAX_DEPTH}.png",
    dpi=300,
    bbox_inches="tight"
)


# Show graph
plt.show()
