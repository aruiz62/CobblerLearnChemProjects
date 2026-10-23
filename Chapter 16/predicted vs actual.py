
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Locate Chapter 16 folder
folder = Path(__file__).resolve().parent

# Load housing data
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Use the same test split as training.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained Random Forest
model = joblib.load(folder / "housing_model.pkl")

# Predict housing prices
predictions = model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, predictions)
print("Random Forest R2:", round(r2, 4))

# Create scatterplot
plt.figure(figsize=(8, 7))

plt.scatter(
    y_test,
    predictions,
    alpha=0.3,
    color="royalblue"
)

# Perfect prediction reference line
plt.plot(
    [0, 5.2],
    [0, 5.2],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual House Price ($100,000s)")
plt.ylabel("Predicted House Price ($100,000s)")
plt.title("Random Forest: Predicted vs Actual Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save graph
plt.savefig(folder / "predicted_vs_actual.png", dpi=300)

plt.show()

print("Scatterplot saved successfully!")
