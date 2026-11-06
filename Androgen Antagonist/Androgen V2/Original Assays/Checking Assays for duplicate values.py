
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

ratio_file = next(folder.glob("*Antagonist_ratio*.csv"))
viability_file = next(folder.glob("*Antagonist_viability*.csv"))

ratio = pd.read_csv(ratio_file, low_memory=False)
viability = pd.read_csv(viability_file, low_memory=False)

for name, df in [("RATIO", ratio), ("VIABILITY", viability)]:
    print(f"\n===== {name} =====")

    print("\nAll column names:")
    for col in df.columns:
        print(repr(col))

    print("\nMissing DTXSIDs:", df["DTXSID"].isna().sum())
    print("Unique DTXSIDs:", df["DTXSID"].nunique())
    print("Duplicate DTXSIDs:", df["DTXSID"].duplicated().sum())

    print("\nHIT CALL values:")
    if "HIT CALL" in df.columns:
        print(df["HIT CALL"].value_counts(dropna=False))
    else:
        print("No HIT CALL column found")

# Check how many chemical identifiers match
ratio_ids = set(ratio["DTXSID"].dropna())
viability_ids = set(viability["DTXSID"].dropna())

print("\n===== DATASET COMPARISON =====")
print("Chemicals in both:", len(ratio_ids & viability_ids))
print("Only in ratio:", len(ratio_ids - viability_ids))
print("Only in viability:", len(viability_ids - ratio_ids))

print("\nAre all columns identical?",
      ratio.columns.tolist() == viability.columns.tolist())

print("Are both datasets completely identical?",
      ratio.equals(viability))
