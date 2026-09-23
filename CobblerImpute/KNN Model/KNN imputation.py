import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
df = pd.read_csv("../alkane_dataset.csv")


# Property we want to predict
property_name = "viscosity"


# Features used to predict viscosity
features = ["carbons", "branch number"]


# Keep only rows where viscosity is known
known = df[property_name].notna()

X = df.loc[known, features]
y = df.loc[known, [property_name]]


# -----------------------------------
# SCALE FEATURES AND VISCOSITY TO 0-1
# -----------------------------------

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y).ravel()


# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.4,
    random_state=42
)


# Test k = 3, 5, and 10
k_values = [3, 5, 10]


for k in k_values:

    # Create KNN model
    model = KNeighborsRegressor(
        n_neighbors=k
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predict test data
    predicted_values = model.predict(
        X_test
    )


    # -----------------------------------
    # CALCULATE MAE
    # -----------------------------------

    mae = mean_absolute_error(
        y_test,
        predicted_values
    )

    print("k =", k)
    print("MAE =", round(mae, 4))
    print()


    # -----------------------------------
    # PREDICTION QUALITY GRAPH
    # -----------------------------------

    plt.figure()

    plt.scatter(
        y_test,
        predicted_values
    )


    # Perfect prediction line
    minimum = min(
        y_test.min(),
        predicted_values.min()
    )

    maximum = max(
        y_test.max(),
        predicted_values.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "--"
    )


    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.title(
        "Prediction Quality - KNN k="
        + str(k)
        + " | MAE = "
        + str(round(mae, 4))
    )


    # Keep axes similar to Cobbler
    plt.xlim(-0.05, 1.25)
    plt.ylim(-0.05, 1.25)


    # Save graph
    plt.savefig(
        "KNN_k" + str(k) + "_prediction_quality.png"
    )


    plt.show()