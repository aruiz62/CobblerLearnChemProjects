import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# =====================================
# SETTINGS
# =====================================

K_NEIGHBORS = 5
MAX_DEPTH = 5


# =====================================
# LOAD DATA
# =====================================

df = pd.read_csv("../alkane_dataset.csv")


# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)


print("Columns in dataset:")
print(df.columns.tolist())


# =====================================
# ENSEMBLE IMPUTATION FUNCTION
# =====================================

def ensemble_impute(data, target):

    # Use carbons and branch number as predictors
    features = ["carbons", "branch_number"]

    # Rows where target value is known
    train_data = data[
        data[target].notna()
        & data["carbons"].notna()
        & data["branch_number"].notna()
    ]

    # Rows where target is missing
    missing_data = data[
        data[target].isna()
        & data["carbons"].notna()
        & data["branch_number"].notna()
    ]

    # Nothing to fill
    if len(missing_data) == 0:
        return data

    X_train = train_data[features]
    y_train = train_data[target]

    X_missing = missing_data[features]


    # -----------------------------
    # Linear Regression
    # -----------------------------

    linear = LinearRegression()

    linear.fit(
        X_train,
        y_train
    )

    linear_prediction = linear.predict(
        X_missing
    )


    # -----------------------------
    # KNN
    # -----------------------------

    k = min(
        K_NEIGHBORS,
        len(X_train)
    )

    knn = KNeighborsRegressor(
        n_neighbors=k
    )

    knn.fit(
        X_train,
        y_train
    )

    knn_prediction = knn.predict(
        X_missing
    )


    # -----------------------------
    # Random Forest
    # -----------------------------

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=MAX_DEPTH,
        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_prediction = rf.predict(
        X_missing
    )


    # -----------------------------
    # Ensemble Average
    # -----------------------------

    ensemble_prediction = (
        linear_prediction
        + knn_prediction
        + rf_prediction
    ) / 3


    # Fill missing values
    data.loc[
        missing_data.index,
        target
    ] = ensemble_prediction

    return data


# =====================================
# FILL MISSING PROPERTY VALUES
# =====================================

# Try to fill every numeric property other than
# the two predictor columns

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    if column not in [
        "carbons",
        "branch_number"
    ]:

        if df[column].isna().any():

            print(
                f"Imputing missing values for: {column}"
            )

            df = ensemble_impute(
                df,
                column
            )


# =====================================
# CORRELATION MATRIX
# =====================================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()


print("\nCorrelation Matrix:")
print(correlation.round(2))


# =====================================
# CREATE HEATMAP
# =====================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

heatmap = ax.imshow(
    correlation,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)


# Column labels
ax.set_xticks(
    np.arange(len(correlation.columns))
)

ax.set_xticklabels(
    correlation.columns,
    rotation=45,
    ha="right"
)


# Row labels
ax.set_yticks(
    np.arange(len(correlation.index))
)

ax.set_yticklabels(
    correlation.index
)


# Put correlation numbers inside squares
for i in range(len(correlation.index)):

    for j in range(len(correlation.columns)):

        value = correlation.iloc[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center"
        )


# Color bar
plt.colorbar(
    heatmap,
    label="Correlation"
)


# Title
plt.title(
    "Correlation Matrix"
)

plt.tight_layout()


# =====================================
# SAVE GRAPH
# =====================================

plt.savefig(
    "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


# Show graph
plt.show()