import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Load the dataset
df = pd.read_csv("alkane_dataset.csv")


# Property we want to predict/impute
property_name = "viscosity"


# Features used to predict viscosity
features = ["carbons", "branch number"]


# Find rows where viscosity is known
known = df[property_name].notna()


# Create X and y using only rows with known viscosity
X = df.loc[known, features]

# Take the logarithm of viscosity
y = np.log(df.loc[known, property_name])


# Split the known data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create the log-linear regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# -------------------------
# IMPUTE MISSING VALUES
# -------------------------

# Find rows where viscosity is missing
missing = df[property_name].isna()

# Make predictions for missing values
missing_predictions_log = model.predict(df.loc[missing, features])

# Convert predictions back from log scale
missing_predictions = np.exp(missing_predictions_log)

# Make a copy of the dataset
imputed_df = df.copy()

# Put predicted values into the missing spots
imputed_df.loc[missing, property_name] = missing_predictions


# Show how many values are still missing
print("Missing values after imputation:")
print(imputed_df.isnull().sum())


# -------------------------
# TEST MODEL ACCURACY
# -------------------------

# Predict the TEST data
predicted_log = model.predict(X_test)

# Convert predicted values back from log scale
predicted_values = np.exp(predicted_log)

# Convert actual test values back from log scale
actual_values = np.exp(y_test)


# Calculate MAE
mae = mean_absolute_error(actual_values, predicted_values)

print("MAE:", round(mae, 2))


# -------------------------
# PREDICTION QUALITY GRAPH
# -------------------------

plt.scatter(actual_values, predicted_values)


# Make the perfect prediction line
minimum = min(actual_values.min(), predicted_values.min())
maximum = max(actual_values.max(), predicted_values.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)


# Label the graph
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Prediction Quality")


# Show the graph
plt.savefig("prediction_quality.png")
plt.show()