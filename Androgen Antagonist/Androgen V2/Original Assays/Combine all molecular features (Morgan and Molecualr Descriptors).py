
from pathlib import Path
import pandas as pd

# STEP 1: Locate our files
folder = Path(__file__).resolve().parent

fingerprints = pd.read_csv(
    folder / "Morgan_Fingerprints.csv"
)

descriptors = pd.read_csv(
    folder / "Molecular_Descriptors.csv"
)

print("Fingerprint chemicals:", len(fingerprints))
print("Descriptor chemicals:", len(descriptors))

# STEP 2: Check for duplicate chemical IDs
print("\nDuplicate fingerprint IDs:",
      fingerprints["DTXSID"].duplicated().sum())

print("Duplicate descriptor IDs:",
      descriptors["DTXSID"].duplicated().sum())

# STEP 3: Combine using DTXSID
# Only merge the five descriptor columns.
# This avoids duplicating the activity labels.

descriptor_columns = [
    "DTXSID", "MW", "LogP", "HBD", "HBA", "TPSA"
]

combined = pd.merge(
    fingerprints,
    descriptors[descriptor_columns],
    on="DTXSID",
    how="inner",
    validate="one_to_one"
)

# STEP 4: Verify activity labels and chemical counts
assert len(combined) == len(fingerprints), \
    "Some fingerprint chemicals were lost during merging!"

assert len(combined) == len(descriptors), \
    "Some descriptor chemicals were lost during merging!"

# STEP 5: Check missing values
print("\n===== COMBINED DATASET RESULTS =====")
print("Total chemicals:", len(combined))
print("Total columns:", len(combined.columns))
print("Molecular features:", len(combined.columns) - 3)
print("Missing values:", combined.isna().sum().sum())
print("Duplicate DTXSIDs:", combined["DTXSID"].duplicated().sum())

assert combined.isna().sum().sum() == 0, \
    "Missing values found in combined dataset!"

# STEP 6: Save new dataset
output = folder / "Androgen_ML_Features.csv"

combined.to_csv(output, index=False)

print("\nFile saved:", output.name)
print("\nFirst 5 chemicals:")
print(combined[[
    "DTXSID", "BLA_Activity", "Viability_Activity",
    "MW", "LogP", "HBD", "HBA", "TPSA"
]].head())
