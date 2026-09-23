import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =====================================
# LOAD TITANIC DATASET
# =====================================

titanic = sns.load_dataset("titanic")


# =====================================
# SHOW FIRST 10 ROWS
# =====================================

print("First 10 rows:")
print(titanic.head(10))


# =====================================
# CHECK MISSING AGE VALUES BEFORE
# =====================================

missing_before = titanic["age"].isna().sum()

print("\nMissing Age values before imputation:")
print(missing_before)


# =====================================
# CALCULATE MEAN AGE
# Uses only known/non-missing ages
# =====================================

mean_age = titanic["age"].mean()

print("\nMean Age:")
print(round(mean_age, 2))


# =====================================
# FILL MISSING AGE VALUES
# =====================================

titanic["age"] = titanic["age"].fillna(mean_age)


# =====================================
# CHECK MISSING AGE VALUES AFTER
# =====================================

missing_after = titanic["age"].isna().sum()

print("\nMissing Age values after imputation:")
print(missing_after)

if missing_after == 0:
    print("Success: There are no missing Age values after imputation.")
else:
    print(f"There are still {missing_after} missing Age values.")


# =====================================
# CREATE CORRELATION MATRIX
# =====================================

numeric_data = titanic.select_dtypes(include="number")

correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))


# =====================================
# FIND TWO STRONGEST CORRELATIONS
# WITH AGE
# =====================================

age_correlations = (
    correlation_matrix["age"]
    .drop("age")
    .abs()
    .sort_values(ascending=False)
)

print("\nTwo features most strongly correlated with Age:")
print(age_correlations.head(2))


# =====================================
# CREATE CORRELATION HEATMAP
# =====================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Titanic Correlation Matrix")

plt.tight_layout()


# =====================================
# SAVE GRAPH AUTOMATICALLY
# =====================================

plt.savefig(
    "titanic_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================
# SHOW GRAPH
# =====================================

plt.show()
