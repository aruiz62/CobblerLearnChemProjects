import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. CREATE NOISY DATA
# ==========================================

np.random.seed(42)

# x-values from 0 to 10
x = np.linspace(0, 10, 50)

# True line: y = 2x + 5
true_y = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, size=len(x))
y = true_y + noise


# ==========================================
# 2. PLOT NOISY DATA + TRUE LINE
# ==========================================

plt.scatter(x, y, label="Noisy Data")
plt.plot(x, true_y, label="True Line: y = 2x + 5")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Noisy Data and True Line")
plt.legend()
plt.show()


# ==========================================
# 3. TRAIN LINEAR REGRESSION MODEL
# ==========================================

# sklearn needs X to be 2D
X = x.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

# Model predictions
predicted_y = model.predict(X)


# Plot results
plt.scatter(x, y, label="Noisy Data")
plt.plot(x, true_y, label="True Line")
plt.plot(x, predicted_y, label="Best-Fit Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("True Line vs Best-Fit Line")
plt.legend()
plt.show()


# ==========================================
# 4. PRINT SLOPE AND INTERCEPT
# ==========================================

print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)

print("\nTrue slope: 2")
print("True intercept: 5")


# ==========================================
# 6. CALCULATE MSE
# ==========================================

def calculate_mse(slope, intercept):

    predicted = slope * x + intercept

    mse = mean_squared_error(y, predicted)

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("MSE:", mse)

    return mse


# Example random point
calculate_mse(3, 4)