
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# STEP 1: Create histograms
df.hist(figsize=(14, 10), bins=30, edgecolor="black")

plt.suptitle("California Housing Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("housing_histograms.png", dpi=300)
plt.show()

print("Histograms saved successfully!")

# STEP 2: Create correlation heatmap
plt.figure(figsize=(12, 9))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("California Housing Correlation Heatmap")
plt.tight_layout()
plt.savefig("housing_heatmap.png", dpi=300)
plt.show()

print("Heatmap saved successfully!")

# STEP 3: Show correlations with house prices
print("\nCorrelation with Housing Prices:")
print(df.corr()["MedHouseVal"].sort_values(ascending=False))
