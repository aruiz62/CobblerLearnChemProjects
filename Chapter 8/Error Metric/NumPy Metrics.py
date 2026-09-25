import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# DATA
# ---------------------------------

actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])


# ---------------------------------
# RESIDUALS
# Predicted - Actual
# ---------------------------------

residuals = predicted - actual


# ---------------------------------
# CALCULATE METRICS WITH NUMPY
# ---------------------------------

mae = np.mean(np.abs(residuals))

mse = np.mean(residuals ** 2)

r2 = 1 - (
    np.sum((actual - predicted) ** 2)
    /
    np.sum((actual - np.mean(actual)) ** 2)
)


print("Metrics calculated with NumPy")
print("--------------------------------")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²:  {r2:.4f}")


# ---------------------------------
# CHECK WITH SCIKIT-LEARN
# ---------------------------------

sklearn_mae = mean_absolute_error(actual, predicted)
sklearn_mse = mean_squared_error(actual, predicted)
sklearn_r2 = r2_score(actual, predicted)

print("\nMetrics calculated with scikit-learn")
print("--------------------------------")
print(f"MAE: {sklearn_mae:.4f}")
print(f"MSE: {sklearn_mse:.4f}")
print(f"R²:  {sklearn_r2:.4f}")


# ---------------------------------
# TABLE OF VALUES
# ---------------------------------

table = pd.DataFrame({
    "Actual": actual,
    "Predicted": predicted,
    "Residual": residuals
})

print("\nActual | Predicted | Residual")
print("--------------------------------")
print(table.to_string(index=False))


# Save table as CSV
table.to_csv(
    "actual_predicted_residual_table.csv",
    index=False
)


# ---------------------------------
# PREDICTED VS ACTUAL PLOT
# ---------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(actual, predicted, s=70)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--",
    label="Ideal Fit"
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "predicted_vs_actual.png",
    dpi=300
)

plt.show()


# ---------------------------------
# RESIDUAL PLOT
# ---------------------------------

plt.figure(figsize=(7, 5))

plt.scatter(actual, residuals, s=70)

plt.axhline(
    y=0,
    linestyle="--",
    label="Zero Error"
)

plt.xlabel("Actual")
plt.ylabel("Residual (Predicted - Actual)")
plt.title("Residual Plot")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "residual_plot.png",
    dpi=300
)

plt.show()
