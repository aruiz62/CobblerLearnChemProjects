
import pandas as pd
from pathlib import Path

# Find the folder containing our original CSV files
folder = Path(__file__).resolve().parent

# Locate the two assays
ratio_file = next(folder.glob("*Antagonist_ratio*.csv"))
viability_file = next(folder.glob("*Antagonist_viability*.csv"))

# Load the datasets
ratio = pd.read_csv(ratio_file, low_memory=False)
viability = pd.read_csv(viability_file, low_memory=False)

# Keep only chemical IDs and activity labels
ratio = ratio[["DTXSID", "HIT CALL"]].copy()
viability = viability[["DTXSID", "HIT CALL"]].copy()

# Rename columns so we can distinguish the assays
ratio.rename(columns={"HIT CALL": "BLA_Activity"}, inplace=True)
viability.rename(columns={"HIT CALL": "Viability_Activity"}, inplace=True)

# Merge using chemical identifiers
merged = pd.merge(
    ratio,
    viability,
    on="DTXSID",
    how="inner",
    validate="one_to_one"
)

# Count each activity combination
comparison = pd.crosstab(
    merged["BLA_Activity"],
    merged["Viability_Activity"]
)

print("\n===== ACTIVITY COMPARISON =====")
print(comparison)

print("\nTotal matched chemicals:", len(merged))

# Calculate chemicals with antagonist activity
# but no detected viability activity
candidates = merged[
    (merged["BLA_Activity"] == "Active") &
    (merged["Viability_Activity"] == "Inactive")
]

print("\nBLA active / Viability inactive:", len(candidates))

# Save the results without changing original files
merged.to_csv(folder / "activity_comparison.csv", index=False)
comparison.to_csv(folder / "activity_summary.csv")

print("\nFiles saved successfully!")
