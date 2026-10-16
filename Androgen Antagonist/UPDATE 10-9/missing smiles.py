import pandas as pd

df = pd.read_csv("AR_BLA_Master_with_EPA_SMILES.csv")

missing_regular = df[df["epa_smiles"].isna()]

print("Missing regular SMILES:", len(missing_regular))

print(
    "Of those, QSAR-ready SMILES available:",
    missing_regular["epa_qsar_ready_smiles"].notna().sum()
)

print(
    "Missing BOTH regular and QSAR-ready SMILES:",
    (
        missing_regular["epa_qsar_ready_smiles"].isna()
    ).sum()
)