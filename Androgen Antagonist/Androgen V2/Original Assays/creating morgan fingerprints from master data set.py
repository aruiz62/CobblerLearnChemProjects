
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# STEP 1: Load master dataset
folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "Master_Androgen_Dataset.csv")

print("Original chemicals:", len(df))

# STEP 2: Set up Morgan fingerprints
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

# STEP 3: Generate fingerprints from QSAR SMILES
fingerprints = []
valid_indices = []

for index, smiles in df["QSAR_READY_SMILES"].items():

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        print("Invalid SMILES at row:", index)
        continue

    fp = generator.GetFingerprintAsNumPy(mol)

    fingerprints.append(fp)
    valid_indices.append(index)

# STEP 4: Create fingerprint columns
fp_columns = [f"Morgan_{i}" for i in range(2048)]

fp_df = pd.DataFrame(
    np.array(fingerprints, dtype=np.uint8),
    columns=fp_columns
)

# STEP 5: Combine fingerprints with activity labels
chemical_info = df.loc[
    valid_indices,
    ["DTXSID", "BLA_Activity", "Viability_Activity"]
].reset_index(drop=True)

final_df = pd.concat(
    [chemical_info, fp_df],
    axis=1
)

# STEP 6: Save results
output = folder / "Morgan_Fingerprints.csv"

final_df.to_csv(output, index=False)

# STEP 7: Check results
print("\n===== MORGAN FINGERPRINT RESULTS =====")
print("Chemicals processed:", len(final_df))
print("Fingerprint bits:", 2048)
print("Total columns:", len(final_df.columns))
print("Missing values:", final_df.isna().sum().sum())
print("File saved:", output.name)

print("\nFirst 5 chemicals:")
print(final_df.iloc[:5, :13])
