
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors

# STEP 1: Load master dataset
folder = Path(__file__).resolve().parent
df = pd.read_csv(folder / "Master_Androgen_Dataset.csv")

print("Original chemicals:", len(df))

# STEP 2: Calculate molecular descriptors
results = []

for _, row in df.iterrows():

    smiles = row["QSAR_READY_SMILES"]
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        print("Invalid SMILES:", row["DTXSID"])
        continue

    results.append({
        "DTXSID": row["DTXSID"],
        "BLA_Activity": row["BLA_Activity"],
        "Viability_Activity": row["Viability_Activity"],
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol)
    })

# STEP 3: Create a new dataset
descriptors_df = pd.DataFrame(results)

# STEP 4: Save the results
output = folder / "Molecular_Descriptors.csv"
descriptors_df.to_csv(output, index=False)

# STEP 5: Check the results
print("\n===== MOLECULAR DESCRIPTOR RESULTS =====")
print("Chemicals processed:", len(descriptors_df))
print("Total columns:", len(descriptors_df.columns))
print("Missing values:", descriptors_df.isna().sum().sum())
print("File saved:", output.name)

print("\nFirst 5 chemicals:")
print(descriptors_df.head())
