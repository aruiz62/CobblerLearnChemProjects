import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# The four datasets
datasets = [
    "dataset_low_scatter.csv",
    "dataset_high_scatter.csv",
    "dataset_positive_deviation.csv",
    "dataset_negative_deviation.csv"
]

for filename in datasets:

    # Load the CSV
    df = pd.read_csv(filename)

    # Makes column names lowercase just in case
    df.columns = df.columns.str.lower().str.strip()

    # Get actual and predicted values
    actual = df["actual"]
    predicted = df["predicted"]

    # Residual = predicted - actual
    residuals = predicted - actual

    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print("\n------------------------------")
    print(filename)
    print("------------------------------")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    # -----------------------------
    # Predicted vs Actual Plot
    # -----------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(actual, predicted)

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "--",
        label="Ideal Fit"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(filename.replace(".csv", "") + " - Predicted vs Actual")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        filename.replace(".csv", "_predicted_vs_actual.png"),
        dpi=300
    )

    plt.show()

    # -----------------------------
    # Residual Plot
    # -----------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(actual, residuals)

    plt.axhline(
        y=0,
        linestyle="--",
        label="Zero Error"
    )

    plt.xlabel("Actual")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.title(filename.replace(".csv", "") + " - Residual Plot")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        filename.replace(".csv", "_residual_plot.png"),
        dpi=300
    )

    plt.show()