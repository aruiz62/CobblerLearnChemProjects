
from pathlib import Path
import pandas as pd

# Locate our files
folder = Path(__file__).resolve().parent
smiles_file = folder / "QSAR SMILES" / "valid_qsar_smiles.csv"
activity_file = folder / "activity_comparison.csv"

# Load the datasets
smiles = pd.read_csv(smiles_file)
activity = pd.read_csv(activity_file)

# Standardize chemical identifiers
for df in [smiles, activity]:
    df["DTXSID"] = (
        df["DTXSID"].astype(str)
        .str.extract(r"(DTXSID\d+)", expand=False)
    )

# Make sure each chemical appears only once
assert smiles["DTXSID"].notna().all()
assert activity["DTXSID"].notna().all()
assert smiles["DTXSID"].is_unique
assert activity["DTXSID"].is_unique

# Merge assay activity with chemical structures
master = activity.merge(
    smiles,
    on="DTXSID",
    how="inner",
    validate="one_to_one"
)

# Verify the results
print("Original assay chemicals:", len(activity))
print("Valid SMILES chemicals:", len(smiles))
print("Chemicals successfully merged:", len(master))
print("Chemicals without matched SMILES:", len(activity) - len(master))

print("\nMaster dataset columns:")
print(master.columns.tolist())

print("\nActivity comparison:")
print(pd.crosstab(
    master["BLA_Activity"],
    master["Viability_Activity"]
))

# Save our master dataset
output = folder / "Master_Androgen_Dataset.csv"
master.to_csv(output, index=False)

print("\nSaved:", output.name)
