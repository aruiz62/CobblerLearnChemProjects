import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors


# Load the fingerprint-ready chemicals
df = pd.read_csv("AR_BLA_Fingerprint_Ready_Chemicals.csv")

descriptor_rows = []

for index, row in df.iterrows():

    # Prefer QSAR-ready SMILES
    smiles = row["epa_qsar_ready_smiles"]

    # Fall back to regular SMILES
    if pd.isna(smiles) or not str(smiles).strip():
        smiles = row["epa_smiles"]

    # Skip if no SMILES
    if pd.isna(smiles) or not str(smiles).strip():
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        continue

    descriptor_rows.append({
        "dtxsid_clean": row["dtxsid_clean"],

        "Molecular_Weight": Descriptors.MolWt(mol),

        "LogP": Crippen.MolLogP(mol),

        "TPSA": rdMolDescriptors.CalcTPSA(mol),

        "H_Bond_Donors": Lipinski.NumHDonors(mol),

        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),

        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),

        "Ring_Count": Lipinski.RingCount(mol),

        "Heavy_Atom_Count": Lipinski.HeavyAtomCount(mol)
    })


descriptors = pd.DataFrame(descriptor_rows)

print("Chemicals with descriptors:", len(descriptors))

print("\nFirst 10:")
print(descriptors.head(10))

print("\nDescriptor summary:")
print(descriptors.describe())

# Save
descriptors.to_csv(
    "AR_BLA_Molecular_Descriptors.csv",
    index=False
)

print("\nSaved AR_BLA_Molecular_Descriptors.csv")