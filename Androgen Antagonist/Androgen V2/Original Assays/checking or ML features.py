
import pandas as pd

df = pd.read_csv("Androgen_ML_Features.csv")

print("Dataset shape:", df.shape)

print("\nLast 15 columns:")
print(df.columns[-15:].tolist())

print("\nMissing values:", df.isnull().sum().sum())

print("\nBLA activity:")
print(df["BLA_Activity"].value_counts())

print("\nViability activity:")
print(df["Viability_Activity"].value_counts())


print("\nBLA vs Viability:")
print(pd.crosstab(
    df["BLA_Activity"],
    df["Viability_Activity"],
    margins=True
))
