
import pandas as pd
from rdkit import Chem
from pathlib import Path

folder = Path(__file__).resolve().parent

# Load the original EPA download
file = folder / "CCD-Batch-Search_2026-10-30_05_22_13.csv"
df = pd.read_csv(file, low_memory=False)

# Check every QSAR-ready SMILES
def check_smiles(smiles):
    if pd.isna(smiles) or not str(smiles).strip():
        return "Missing"

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return "Invalid"

    return "Valid"

df["SMILES_Status"] = df["QSAR_READY_SMILES"].apply(check_smiles)

# Count each result
print("===== SMILES VALIDATION =====")
print("Total chemicals:", len(df))
print(df["SMILES_Status"].value_counts())

# Save all results, including missing structures
df.to_csv(folder / "smiles_validation_results.csv", index=False)

# Save only valid structures for later modeling
valid = df[df["SMILES_Status"] == "Valid"].copy()
valid.to_csv(folder / "valid_qsar_smiles.csv", index=False)

# Save missing and invalid structures for documentation
excluded = df[df["SMILES_Status"] != "Valid"].copy()
excluded.to_csv(folder / "missing_invalid_smiles.csv", index=False)

print("\nValid chemicals for modeling:", len(valid))
print("Missing or invalid structures:", len(excluded))
print("\nFiles saved successfully!")
