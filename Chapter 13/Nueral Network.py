
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Save graphs in the same folder as this Python file
output_folder = Path(__file__).resolve().parent

# STEP 1: Create the chemistry dataset
data = {
    "Compound": [
        "Methane", "Water", "Propane", "Ethanol",
        "Formic Acid", "Acetic Acid", "Butane",
        "Acetone", "Benzene", "Toluene", "Octane"
    ],
    "MW": [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    "BoilingPoint": [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}

df = pd.DataFrame(data)

print("Chemistry Dataset:")
print(df)

# X = Input (molecular weight)
# y = Target (boiling point)
X = df[["MW"]]
y = df["BoilingPoint"]

# STEP 2: Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict boiling points
y_pred = model.predict(X)

# Calculate model performance
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nLinear Regression Results:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# GRAPH 1: Actual data and regression line
plt.figure(figsize=(8, 5))

plt.scatter(X["MW"], y, color="blue", label="Actual Data")

sorted_indices = np.argsort(X["MW"].values)

plt.plot(
    X["MW"].values[sorted_indices],
    y_pred[sorted_indices],
    color="red",
    label="Linear Regression"
)

plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression: Molecular Weight vs Boiling Point")
plt.legend()

# Display MAE, MSE, and R² on the graph
metrics_text = (
    f"MAE: {mae:.2f}\n"
    f"MSE: {mse:.2f}\n"
    f"R²: {r2:.4f}"
)

plt.gca().text(
    0.98, 0.05,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9)
)

plt.grid(True)

plt.savefig(output_folder / "linear_regression.png", dpi=300, bbox_inches="tight")
plt.close()

# GRAPH 2: Residuals
residuals = y - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, color="purple")
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Boiling Point (°C)")
plt.ylabel("Residuals (°C)")
plt.title("Linear Regression Residual Plot")
plt.grid(True)

plt.savefig(output_folder / "linear_regression_residuals.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nBoth graphs have been saved successfully!")
df = pd.DataFrame(data)
df.to_csv(output_folder / "chemistry_dataset.csv", index=False)
print("Chemistry dataset saved!")

# STEP 3: TRAIN A NEURAL NETWORK

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Scale molecular weights and boiling points
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

# Create the neural network
nn_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=5000,
    early_stopping=False,
    tol=0,
    n_iter_no_change=5001,
    random_state=42
)

# Train the neural network
nn_model.fit(X_scaled, y_scaled)

# Predict boiling points
nn_scaled_predictions = nn_model.predict(X_scaled)
nn_predictions = y_scaler.inverse_transform(
    nn_scaled_predictions.reshape(-1, 1)
).ravel()

# Calculate performance
nn_mae = mean_absolute_error(y, nn_predictions)
nn_mse = mean_squared_error(y, nn_predictions)
nn_r2 = r2_score(y, nn_predictions)

print("\nNEURAL NETWORK RESULTS")
print(f"MAE: {nn_mae:.2f}")
print(f"MSE: {nn_mse:.2f}")
print(f"R²: {nn_r2:.4f}")
print(f"Training epochs completed: {nn_model.n_iter_}")

# Plot actual data and neural network predictions
X_plot = np.linspace(X["MW"].min(), X["MW"].max(), 200).reshape(-1, 1)

X_plot_scaled = x_scaler.transform(
    pd.DataFrame(X_plot, columns=["MW"])
)

nn_plot_scaled = nn_model.predict(X_plot_scaled)
nn_plot = y_scaler.inverse_transform(
    nn_plot_scaled.reshape(-1, 1)
).ravel()

plt.figure(figsize=(9, 6))

plt.scatter(X["MW"], y, color="blue", label="Actual Data")
plt.plot(X_plot, nn_plot, color="green", label="Neural Network")

plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Neural Network: Molecular Weight vs Boiling Point")
plt.legend()
plt.grid(True)

# Display metrics on graph
metrics_text = (
    f"MAE: {nn_mae:.2f}\n"
    f"MSE: {nn_mse:.2f}\n"
    f"R²: {nn_r2:.4f}\n"
    f"Epochs: {nn_model.n_iter_}"
)

plt.gca().text(
    0.98, 0.05,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    ha="right",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9)
)

plt.savefig(
    output_folder / "neural_network.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nNeural network graph saved successfully!")


# STEP 4: TEST BOTH MODELS USING BUTANE

# Remove Butane from the training dataset
train_df = df[df["Compound"] != "Butane"].copy()
test_df = df[df["Compound"] == "Butane"].copy()

X_train = train_df[["MW"]]
y_train = train_df["BoilingPoint"]

X_test = test_df[["MW"]]
y_test = test_df["BoilingPoint"].iloc[0]

# Retrain linear regression without Butane
linear_test_model = LinearRegression()
linear_test_model.fit(X_train, y_train)

linear_prediction = linear_test_model.predict(X_test)[0]

# Scale training data for neural network
x_scaler_test = StandardScaler()
y_scaler_test = StandardScaler()

X_train_scaled = x_scaler_test.fit_transform(X_train)

y_train_scaled = y_scaler_test.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

X_test_scaled = x_scaler_test.transform(X_test)

# Retrain neural network without Butane
nn_test_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=5000,
    early_stopping=False,
    tol=0,
    n_iter_no_change=5001,
    random_state=42
)

nn_test_model.fit(X_train_scaled, y_train_scaled)

nn_prediction_scaled = nn_test_model.predict(X_test_scaled)

nn_prediction = y_scaler_test.inverse_transform(
    nn_prediction_scaled.reshape(-1, 1)
).ravel()[0]

# Calculate prediction errors
linear_error = linear_prediction - y_test
nn_error = nn_prediction - y_test

print("\nSTEP 4: BUTANE PREDICTION RESULTS")
print(f"Actual Butane BP: {y_test:.2f} °C")
print(f"Linear Regression Prediction: {linear_prediction:.2f} °C")
print(f"Neural Network Prediction: {nn_prediction:.2f} °C")

print(f"Linear Regression Error: {linear_error:.2f} °C")
print(f"Neural Network Error: {nn_error:.2f} °C")

print(f"Linear Absolute Error: {abs(linear_error):.2f} °C")
print(f"Neural Network Absolute Error: {abs(nn_error):.2f} °C")

# Determine which model predicted better
if abs(linear_error) < abs(nn_error):
    print("Linear Regression predicted Butane more accurately.")
elif abs(nn_error) < abs(linear_error):
    print("Neural Network predicted Butane more accurately.")
else:
    print("Both models had the same absolute prediction error.")

print(f"Neural Network Epochs: {nn_test_model.n_iter_}")

# GRAPH: Compare actual and predicted boiling points
labels = ["Actual Butane", "Linear Regression", "Neural Network"]

values = [y_test, linear_prediction, nn_prediction]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    labels,
    values,
    color=["blue", "red", "green"]
)

plt.axhline(0, color="black", linewidth=1)

plt.ylabel("Boiling Point (°C)")
plt.title("Butane: Actual vs Predicted Boiling Points")

# Display values above or below each bar
for bar, value in zip(bars, values):
    plt.annotate(
        f"{value:.2f} °C",
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 5 if value >= 0 else -15),
        textcoords="offset points",
        ha="center"
    )

# Display prediction errors on graph
error_text = (
    f"Linear Absolute Error: {abs(linear_error):.2f} °C\n"
    f"Neural Network Absolute Error: {abs(nn_error):.2f} °C"
)

plt.gca().text(
    0.98, 0.98,
    error_text,
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9)
)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save graph
plt.savefig(
    output_folder / "butane_prediction_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nButane prediction comparison graph saved!")


# STEP 5: COMPARE BOTH MODELS

# Predictions using the original full dataset
linear_predictions = model.predict(X)
neural_predictions = nn_predictions

# Calculate metrics for linear regression
linear_mae = mean_absolute_error(y, linear_predictions)
linear_mse = mean_squared_error(y, linear_predictions)
linear_r2 = r2_score(y, linear_predictions)

# Calculate metrics for neural network
neural_mae = mean_absolute_error(y, neural_predictions)
neural_mse = mean_squared_error(y, neural_predictions)
neural_r2 = r2_score(y, neural_predictions)

# Print comparison results
print("\nSTEP 5: MODEL COMPARISON")

print("\nLinear Regression:")
print(f"MAE: {linear_mae:.2f}")
print(f"MSE: {linear_mse:.2f}")
print(f"R²: {linear_r2:.4f}")

print("\nNeural Network:")
print(f"MAE: {neural_mae:.2f}")
print(f"MSE: {neural_mse:.2f}")
print(f"R²: {neural_r2:.4f}")
print(f"Training Epochs: {nn_model.n_iter_}")

# Create smooth lines for both models
X_smooth = np.linspace(
    X["MW"].min(),
    X["MW"].max(),
    300
).reshape(-1, 1)

X_smooth_df = pd.DataFrame(X_smooth, columns=["MW"])

linear_smooth = model.predict(X_smooth_df)

neural_smooth_scaled = nn_model.predict(
    x_scaler.transform(X_smooth_df)
)

neural_smooth = y_scaler.inverse_transform(
    neural_smooth_scaled.reshape(-1, 1)
).ravel()

# GRAPH 1: Compare both models
plt.figure(figsize=(10, 6))

plt.scatter(
    X["MW"], y,
    color="black",
    s=65,
    label="Actual Data",
    zorder=3
)

plt.plot(
    X_smooth.ravel(),
    linear_smooth,
    color="red",
    label="Linear Regression"
)

plt.plot(
    X_smooth.ravel(),
    neural_smooth,
    color="blue",
    label="Neural Network"
)

plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression vs Neural Network")
plt.legend()
plt.grid(True, alpha=0.3)

# Add model metrics to graph
metrics_text = (
    "LINEAR REGRESSION\n"
    f"MAE: {linear_mae:.2f}\n"
    f"MSE: {linear_mse:.2f}\n"
    f"R²: {linear_r2:.4f}\n\n"
    "NEURAL NETWORK\n"
    f"MAE: {neural_mae:.2f}\n"
    f"MSE: {neural_mse:.2f}\n"
    f"R²: {neural_r2:.4f}\n"
    f"Epochs: {nn_model.n_iter_}"
)

plt.gca().text(
    0.98, 0.03,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    ha="right",
    va="bottom",
    bbox=dict(
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)

plt.tight_layout()

plt.savefig(
    output_folder / "model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# GRAPH 2: ReLU Activation Function

# Apply ReLU to the molecular weight inputs
mw_values = np.sort(X["MW"].to_numpy())
relu_values = np.maximum(0, mw_values)

plt.figure(figsize=(9, 5))

plt.plot(
    mw_values,
    relu_values,
    color="purple",
    marker="o",
    label="ReLU Activation"
)

plt.axhline(0, color="black", linestyle="--")

plt.xlabel("Molecular Weight")
plt.ylabel("ReLU Output")
plt.title("ReLU Activation Function Applied to Molecular Weight")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_folder / "relu_activation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nBoth Step 5 graphs saved successfully!")


# STEP 6: EXPERIMENT WITH NEURAL NETWORK ARCHITECTURE

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Use the original dataset
X_step6 = df[["MW"]]
y_step6 = df["BoilingPoint"]

# Scale input and output
x_scaler6 = StandardScaler()
y_scaler6 = StandardScaler()

X_scaled6 = x_scaler6.fit_transform(X_step6)

y_scaled6 = y_scaler6.fit_transform(
    y_step6.to_numpy().reshape(-1, 1)
).ravel()

# Original model: 2 layers, 10 neurons each
original_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=5000,
    early_stopping=False,
    tol=0,
    n_iter_no_change=5001,
    random_state=42
)

original_model.fit(X_scaled6, y_scaled6)

original_scaled = original_model.predict(X_scaled6)

original_predictions = y_scaler6.inverse_transform(
    original_scaled.reshape(-1, 1)
).ravel()

original_mae = mean_absolute_error(y_step6, original_predictions)
original_mse = mean_squared_error(y_step6, original_predictions)
original_r2 = r2_score(y_step6, original_predictions)

print("\nORIGINAL NEURAL NETWORK")
print(f"MAE: {original_mae:.2f}")
print(f"MSE: {original_mse:.2f}")
print(f"R²: {original_r2:.4f}")
print(f"Epochs: {original_model.n_iter_}")

# Repeat until the user decides to stop
experiment_number = 1

while True:

    print("\nSTEP 6: CREATE A NEW NEURAL NETWORK")

    # Ask for number of layers
    while True:
        try:
            num_layers = int(input("How many hidden layers (1-4)? "))
            if 1 <= num_layers <= 4:
                break
            print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a whole number.")

    # Ask for neurons in each layer
    neurons = []

    for i in range(num_layers):
        while True:
            try:
                num_neurons = int(
                    input(f"How many neurons in layer {i + 1} (1-10)? ")
                )
                if 1 <= num_neurons <= 10:
                    neurons.append(num_neurons)
                    break
                print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a whole number.")

    architecture = tuple(neurons)

    print(f"\nTraining neural network with layers: {architecture}")

    # Create new model
    new_model = MLPRegressor(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="adam",
        max_iter=5000,
        early_stopping=False,
        tol=0,
        n_iter_no_change=5001,
        random_state=42
    )

    # Train model
    new_model.fit(X_scaled6, y_scaled6)

    # Predict
    new_scaled = new_model.predict(X_scaled6)

    new_predictions = y_scaler6.inverse_transform(
        new_scaled.reshape(-1, 1)
    ).ravel()

    # Calculate metrics
    new_mae = mean_absolute_error(y_step6, new_predictions)
    new_mse = mean_squared_error(y_step6, new_predictions)
    new_r2 = r2_score(y_step6, new_predictions)

    print("\nNEW NEURAL NETWORK RESULTS")
    print(f"Architecture: {architecture}")
    print(f"MAE: {new_mae:.2f}")
    print(f"MSE: {new_mse:.2f}")
    print(f"R²: {new_r2:.4f}")
    print(f"Epochs: {new_model.n_iter_}")

    # Smooth prediction lines
    X_plot6 = np.linspace(
        X_step6["MW"].min(),
        X_step6["MW"].max(),
        300
    ).reshape(-1, 1)

    X_plot6_df = pd.DataFrame(X_plot6, columns=["MW"])
    X_plot6_scaled = x_scaler6.transform(X_plot6_df)

    original_plot_scaled = original_model.predict(X_plot6_scaled)
    new_plot_scaled = new_model.predict(X_plot6_scaled)

    original_plot = y_scaler6.inverse_transform(
        original_plot_scaled.reshape(-1, 1)
    ).ravel()

    new_plot = y_scaler6.inverse_transform(
        new_plot_scaled.reshape(-1, 1)
    ).ravel()

    # GRAPH: Compare both neural networks
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        X_step6["MW"],
        y_step6,
        color="black",
        s=65,
        label="Actual Data",
        zorder=3
    )

    ax.plot(
        X_plot6.ravel(),
        original_plot,
        color="blue",
        label="Original NN (10, 10)"
    )

    ax.plot(
        X_plot6.ravel(),
        new_plot,
        color="green",
        label=f"New NN {architecture}"
    )

    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("Boiling Point (°C)")
    ax.set_title("Original vs New Neural Network")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Show metrics on graph
    metrics_text = (
        "ORIGINAL NN (10, 10)\n"
        f"MAE: {original_mae:.2f}\n"
        f"MSE: {original_mse:.2f}\n"
        f"R²: {original_r2:.4f}\n\n"
        f"NEW NN {architecture}\n"
        f"MAE: {new_mae:.2f}\n"
        f"MSE: {new_mse:.2f}\n"
        f"R²: {new_r2:.4f}"
    )

    ax.text(
        1.02, 0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    fig.tight_layout()

    # Save each experiment separately
    architecture_name = "_".join(map(str, architecture))

    filename = (
        f"neural_network_experiment_"
        f"{experiment_number}_{architecture_name}.png"
    )

    fig.savefig(
        output_folder / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nGraph saved: {filename}")

    # Compare model performance
    if new_mae < original_mae:
        print("New model has a lower training MAE.")
    elif new_mae > original_mae:
        print("Original model has a lower training MAE.")
    else:
        print("Both models have the same training MAE.")

    # Ask whether to repeat
    again = input("\nTry another neural network? (yes/no): ").strip().lower()

    if again not in ["yes", "y"]:
        print("\nStep 6 completed!")
        break

    experiment_number += 1
