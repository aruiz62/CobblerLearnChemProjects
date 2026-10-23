
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_california_housing

# Find Chapter 16 folder
folder = Path(__file__).resolve().parent

# Load the trained Random Forest model
model = joblib.load(folder / "housing_model.pkl")

# Load housing feature names
housing = fetch_california_housing()
features = housing.feature_names

# Get feature importance values
importance = model.feature_importances_

# Create a table
df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

df = df.sort_values("Importance", ascending=False)

print("Random Forest Feature Importance:")
print(df.to_string(index=False))

# Save results to CSV
df.to_csv(folder / "feature_importance.csv", index=False)

# Create graph
plt.figure(figsize=(10, 6))

plt.barh(df["Feature"], df["Importance"], color="steelblue")
plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.ylabel("Housing Feature")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

# Save graph
plt.savefig(folder / "feature_importance.png", dpi=300)

plt.show()

print("Feature importance graph saved successfully!")
