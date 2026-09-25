import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# =============================
# LOAD DATA
# =============================

df = pd.read_csv("alkane_dataset.csv")

print(df.columns.tolist())


# =============================
# COLUMN NAMES
# =============================

carbon_col = "carbons"
branch_col = "branch number"

properties = [
    "heat capacity",
    "thermal conductivity"
]


# =============================
# SETTINGS
# =============================

K = 5
DEPTH = 5


# =============================
# LOOP THROUGH BOTH PROPERTIES
# =============================

for target in properties:

    print("\n============================")
    print(target.upper())
    print("============================")

    # Keep rows that have the values we need
    data = df.dropna(
        subset=[carbon_col, branch_col, target]
    ).copy()

    X = data[[carbon_col, branch_col]]
    y = data[target]

    # Log transform
    y_log = np.log1p(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_log,
        test_size=0.20,
        random_state=42
    )

    # Convert actual values back
    actual = np.expm1(y_test)

    carbons = X_test[carbon_col]


    # =============================
    # LOG-LINEAR MODEL
    # =============================

    linear = LinearRegression()

    linear.fit(
        X_train,
        y_train
    )

    linear_log_pred = linear.predict(X_test)

    linear_pred = np.expm1(
        linear_log_pred
    )


    # =============================
    # KNN MODEL
    # =============================

    knn = KNeighborsRegressor(
        n_neighbors=K
    )

    knn.fit(
        X_train,
        y_train
    )

    knn_log_pred = knn.predict(X_test)

    knn_pred = np.expm1(
        knn_log_pred
    )


    # =============================
    # RANDOM FOREST
    # =============================

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=DEPTH,
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_log_pred = rf.predict(X_test)

    rf_pred = np.expm1(
        rf_log_pred
    )


    # =============================
    # ENSEMBLE
    # =============================

    ensemble_log_pred = (
        linear_log_pred
        + knn_log_pred
        + rf_log_pred
    ) / 3

    ensemble_pred = np.expm1(
        ensemble_log_pred
    )


    # =============================
    # STORE ALL 4 MODELS
    # =============================

    models = {
        "Log-Linear Regression": linear_pred,
        "KNN": knn_pred,
        "Random Forest": rf_pred,
        "Ensemble Model": ensemble_pred
    }


    # =============================
    # MAKE A GRAPH FOR EACH MODEL
    # =============================

    for model_name, predicted in models.items():

        # MAE
        mae = mean_absolute_error(
            actual,
            predicted
        )

        # Relative error
        relative_error = (
            (predicted - actual)
            / actual
        ) * 100

        print(
            model_name,
            "MAE =",
            round(mae, 4)
        )


        # =============================
        # GRAPH
        # =============================

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            carbons,
            relative_error
        )

        # Zero error
        plt.axhline(
            y=0,
            linestyle="--",
            label="0% Error"
        )

        # +100% catastrophic error
        plt.axhline(
            y=100,
            linestyle=":"
        )

        # -100% catastrophic error
        plt.axhline(
            y=-100,
            linestyle=":"
        )

        plt.xlabel("Carbons")

        plt.ylabel(
            "Relative Error (%)"
        )

        plt.title(
            f"{model_name}\n"
            f"{target.title()} - Bias vs. Carbons"
        )

        plt.legend()

        plt.tight_layout()


        # =============================
        # SAVE EACH GRAPH
        # =============================

        clean_property = (
            target
            .replace(" ", "_")
        )

        clean_model = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        filename = (
            f"{clean_property}_"
            f"{clean_model}_"
            f"bias_vs_carbons.png"
        )

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()