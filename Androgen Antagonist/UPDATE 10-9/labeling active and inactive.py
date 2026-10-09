
import pandas as pd
from pathlib import Path

# Load the master dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "AR_BLA_Master_Dataset.csv")

# Clean activity labels
df["hit_call_antagonist"] = (
    df["hit_call_antagonist"].astype("string").str.strip()
)
df["hit_call_viability"] = (
    df["hit_call_viability"].astype("string").str.strip()
)

# Compare both assay results
comparison = pd.crosstab(
    df["hit_call_antagonist"],
    df["hit_call_viability"]
)

print("\nANTAGONIST VS VIABILITY")
print(comparison.to_string())

# Find potential androgen antagonists
candidates = df[
    (df["hit_call_antagonist"] == "Active") &
    (df["hit_call_viability"] == "Inactive")
].copy()

print("\nPOTENTIAL ANDROGEN ANTAGONISTS")
print("Total candidates:", len(candidates))

# Show chemical names and results
columns = [
    "preferred_name_antagonist",
    "hit_call_antagonist",
    "hit_call_viability"
]

print(candidates[columns].head(20).to_string(index=False))

# Save filtered dataset
candidates.to_csv(
    folder / "ActiveandInactive_AR_Antagonists.csv",
    index=False
)

print("\nFiltered dataset saved successfully!")
