import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("AR_antagonist_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset size:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

hit_counts = df["HIT CALL"].value_counts()

print("\nHIT CALL counts:")
print(hit_counts)

print("\nHIT CALL percentages:")
print((hit_counts / len(df) * 100).round(2))

hit_counts.plot(kind="bar")
plt.title("Active vs Inactive Chemicals in AR Antagonist Assay")
plt.xlabel("HIT CALL")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("active_inactive_chart.png")
plt.show()