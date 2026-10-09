
import pandas as pd
from pathlib import Path

# STEP 1: Show available CSV files
print("CSV files in folder:")
for file in Path(".").glob("*.csv"):
    print(file.name)

# STEP 2: Enter your actual filenames
antagonist_file = "Assay List TOX21_AR_BLA_Antagonist_ratio-2026-10-09 ALL CHEMICALS.csv"
viability_file = "Assay List TOX21_AR_BLA_Antagonist_viability-2026-10-09 ALL CHEMICALS.csv"

# STEP 3: Load datasets
ant = pd.read_csv(antagonist_file, dtype=str)
via = pd.read_csv(viability_file, dtype=str)

# Clean column names
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df

ant = clean_columns(ant)
via = clean_columns(via)

print("\nAntagonist rows:", len(ant))
print("Viability rows:", len(via))

print("\nAntagonist columns:", ant.columns.tolist())
print("\nViability columns:", via.columns.tolist())

# STEP 4: Find a shared chemical identifier
possible_ids = [
    "dtxsid",
    "dsstox_substance_id",
    "casrn",
    "cas_number",
    "chemical_name",
    "substance_name"
]

chemical_id = next(
    (col for col in possible_ids
     if col in ant.columns and col in via.columns),
    None
)

if chemical_id is None:
    raise ValueError(
        "No matching chemical ID column found. "
        "Check the column names printed above."
    )

print("\nMatching chemicals using:", chemical_id)

# Remove rows without chemical IDs
ant = ant.dropna(subset=[chemical_id]).copy()
via = via.dropna(subset=[chemical_id]).copy()

ant[chemical_id] = ant[chemical_id].str.strip()
via[chemical_id] = via[chemical_id].str.strip()

# Check for duplicate chemical IDs
if ant[chemical_id].duplicated().any():
    raise ValueError(
        "Antagonist file has duplicate chemical IDs. "
        "Check samples before merging."
    )

if via[chemical_id].duplicated().any():
    raise ValueError(
        "Viability file has duplicate chemical IDs. "
        "Check samples before merging."
    )

# STEP 5: Combine both datasets
master = pd.merge(
    ant,
    via,
    on=chemical_id,
    how="left",
    suffixes=("_antagonist", "_viability"),
    indicator=True,
    validate="one_to_one"
)

# Identify which chemicals have viability results
master["viability_data_available"] = (
    master["_merge"] == "both"
)

master = master.drop(columns="_merge")

# STEP 6: Display results
print("\nMASTER DATASET RESULTS")
print("Total antagonist chemicals:", len(master))

print(
    "Chemicals with matching viability data:",
    master["viability_data_available"].sum()
)

print(
    "Chemicals missing viability data:",
    (~master["viability_data_available"]).sum()
)

print(master.head(10).to_string())

# STEP 7: Save master dataset
master.to_csv(
    "AR_BLA_Master_Dataset.csv",
    index=False
)

print("\nMaster dataset saved successfully!")
