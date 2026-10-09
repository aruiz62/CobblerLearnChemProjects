
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ==========================================
# CHAPTER 11: GRADIENT DESCENT
# ==========================================

# Create folder for saved graphs
folder = Path.cwd() / "Chapter11_Graphs"
folder.mkdir(parents=True, exist_ok=True)

# ==========================================
# STEP 1 & 2: GENERATE NOISY DATA
# ==========================================

np.random.seed(42)

# Generate x values
x = np.linspace(0, 10, 50)

# True equation: y = 2x + 5
true_y = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, len(x))
y = true_y + noise

# Graph 1: Noisy Data
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.scatter(x, y, color="blue", label="Noisy Data")
ax1.plot(x, true_y, color="red", label="True Line")

ax1.set_title("Noisy Data: y = 2x + 5")
ax1.set_xlabel("X Values")
ax1.set_ylabel("Y Values")
ax1.legend()
ax1.grid(alpha=0.3)

fig1.tight_layout()
fig1.savefig(folder / "Chapter11_Noisy_Data.png", dpi=300)

# ==========================================
# STEP 3: TRAIN LINEAR REGRESSION
# ==========================================

X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

# Model predictions
predicted_y = model.predict(X)

# Learned parameters
slope = float(model.coef_[0])
intercept = float(model.intercept_)

# ==========================================
# STEP 4 & 5: RESULTS AND REGRESSION
# ==========================================

print("LINEAR REGRESSION RESULTS")
print("-----------------------------")
print(f"True Slope: 2")
print(f"Predicted Slope: {slope:.4f}")
print(f"True Intercept: 5")
print(f"Predicted Intercept: {intercept:.4f}")

print("\nANALYSIS")
print("-----------------------------")
print(f"Slope Difference: {abs(2 - slope):.4f}")
print(f"Intercept Difference: {abs(5 - intercept):.4f}")

# Graph 2: Regression
fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(x, y, color="blue", label="Noisy Data")

ax2.plot(
    x, true_y,
    color="red",
    linestyle="--",
    label="True Line"
)

ax2.plot(
    x, predicted_y,
    color="green",
    linewidth=2,
    label="Best-Fit Line"
)

ax2.set_title("Linear Regression: True vs Predicted")
ax2.set_xlabel("X Values")
ax2.set_ylabel("Y Values")
ax2.legend()
ax2.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig(folder / "Chapter11_Regression.png", dpi=300)

# ==========================================
# STEP 6: CALCULATE MSE
# ==========================================

# Function for Mean Squared Error
def calculate_mse(test_slope, test_intercept):

    # Calculate predictions
    predictions = test_slope * x + test_intercept

    # Calculate squared errors
    squared_errors = (y - predictions) ** 2

    # Calculate average squared error
    mse = np.mean(squared_errors)

    return mse


# CHANGE THESE VALUES TO EXPERIMENT
test_slope = 2.0
test_intercept = 5.0

# Calculate MSE using chosen values
test_mse = calculate_mse(
    test_slope,
    test_intercept
)

# Calculate MSE using best-fit parameters
best_mse = calculate_mse(
    slope,
    intercept
)

print("\nSTEP 6: MSE RESULTS")
print("-----------------------------")
print(f"Test Slope: {test_slope}")
print(f"Test Intercept: {test_intercept}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Best-Fit MSE: {best_mse:.4f}")

# ==========================================
# STEP 7: CREATE LOSS LANDSCAPE
# ==========================================

# Generate slope and intercept ranges
slope_values = np.linspace(0, 4, 150)
intercept_values = np.linspace(-1, 11, 150)

# Create a grid
S, I = np.meshgrid(
    slope_values,
    intercept_values
)

# Calculate MSE at every grid point
loss = np.zeros_like(S)

for row in range(S.shape[0]):
    for col in range(S.shape[1]):

        loss[row, col] = calculate_mse(
            S[row, col],
            I[row, col]
        )

# Graph 3: Loss Landscape
fig3, ax3 = plt.subplots(figsize=(9, 7))

# Yellow = Low MSE
# Purple = High MSE
contour = ax3.contourf(
    S, I, loss,
    levels=50,
    cmap="plasma_r"
)

fig3.colorbar(
    contour,
    ax=ax3,
    label="Mean Squared Error (MSE)"
)

# Mark best-fit parameters
ax3.scatter(
    slope,
    intercept,
    color="black",
    marker="x",
    s=150,
    linewidths=3,
    label="Best-Fit Parameters"
)

# Mark the parameters you tested
ax3.scatter(
    test_slope,
    test_intercept,
    color="cyan",
    marker="o",
    s=70,
    edgecolors="black",
    label="Test Parameters"
)

ax3.set_title("Loss Landscape: Mean Squared Error")
ax3.set_xlabel("Slope")
ax3.set_ylabel("Intercept")
ax3.legend()

fig3.tight_layout()

fig3.savefig(
    folder / "Chapter11_Loss_Landscape.png",
    dpi=300
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\nFINAL RESULTS")
print("-----------------------------")
print(f"Best-Fit Slope: {slope:.4f}")
print(f"Best-Fit Intercept: {intercept:.4f}")
print(f"Best-Fit MSE: {best_mse:.4f}")

print("\nSAVED GRAPHS")
print("-----------------------------")

for file in sorted(folder.glob("*.png")):
    print(file.name)

print("\nFolder Location:")
print(folder.resolve())

# Display all 3 graphs after saving
plt.show()
