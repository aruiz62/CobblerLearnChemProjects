
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load California Housing dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Four regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Support Vector": make_pipeline(
        StandardScaler(),
        SVR()
    )
}

results = {}

# Train and test each model
for name, model in models.items():
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    results[name] = r2

    print(f"{name} R2: {r2:.4f}")
    print()

# Save results
results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "R2": list(results.values())
})

folder = Path(__file__).parent

results_df.to_csv(
    folder / "model_comparison.csv",
    index=False
)

# Create comparison graph
plt.figure(figsize=(10, 6))

bars = plt.bar(
    results.keys(),
    results.values(),
    color=["skyblue", "orange", "green", "purple"]
)

plt.title("Regression Model Comparison")
plt.xlabel("Regression Model")
plt.ylabel("R² Score")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    folder / "model_comparison.png",
    dpi=300
)

plt.show()

# Find best model
best_model = max(results, key=results.get)

print("Best Model:", best_model)
print("Best R2 Score:", round(results[best_model], 4))
