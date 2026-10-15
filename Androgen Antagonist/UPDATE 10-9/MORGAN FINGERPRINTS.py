
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# STEP 1: Load master dataset with EPA SMILES
df = pd.read_csv("AR_BLA_Master_with_EPA_SMILES.csv")

print("Original chemicals:", len(df))

# STEP 2: Set up Morgan fingerprints
# Radius 2 = ECFP4-style fingerprint
# 2048 bits = 2048 molecular features
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

# STEP 3: Create output file
fingerprint_file = "AR_BLA_Morgan_Fingerprints.csv"

fingerprint_columns = [
    f"FP_{i}" for i in range(2048)
]

valid_indices = []
smiles_used = []
smiles_sources = []
invalid_count = 0

# STEP 4: Generate fingerprints
with open(fingerprint_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        ["source_row", "dtxsid_clean"] + fingerprint_columns
    )

    for index, row in df.iterrows():

        # Prefer standardized QSAR-ready SMILES
        smiles = row["epa_qsar_ready_smiles"]
        source = "QSAR_READY"

        # Use regular SMILES if QSAR-ready is missing
        if pd.isna(smiles) or not str(smiles).strip():
            smiles = row["epa_smiles"]
            source = "REGULAR"

        # Skip chemicals without structures
        if pd.isna(smiles) or not str(smiles).strip():
            continue

        smiles = str(smiles).strip()

        try:
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                invalid_count += 1
                continue

            # Generate Morgan fingerprint
            fingerprint = generator.GetFingerprintAsNumPy(mol)

            writer.writerow(
                [index, row["dtxsid_clean"]] +
                fingerprint.tolist()
            )

            valid_indices.append(index)
            smiles_used.append(smiles)
            smiles_sources.append(source)

        except Exception as e:
            invalid_count += 1
            print("Could not process row", index, e)

# STEP 5: Save chemicals used for fingerprinting
clean_df = df.loc[valid_indices].copy()

clean_df.insert(0, "source_row", valid_indices)
clean_df["fingerprint_smiles"] = smiles_used
clean_df["fingerprint_smiles_source"] = smiles_sources

clean_df.to_csv(
    "AR_BLA_Fingerprint_Ready_Chemicals.csv",
    index=False
)

# STEP 6: Show results
print("\n===== MORGAN FINGERPRINT RESULTS =====")
print("Original dataset:", len(df))
print("Chemicals with valid fingerprints:", len(valid_indices))
print("Chemicals excluded:", len(df) - len(valid_indices))
print("Invalid SMILES:", invalid_count)
print("Fingerprint size:", 2048, "bits")

print("\nFiles saved:")
print("AR_BLA_Morgan_Fingerprints.csv")
print("AR_BLA_Fingerprint_Ready_Chemicals.csv")
