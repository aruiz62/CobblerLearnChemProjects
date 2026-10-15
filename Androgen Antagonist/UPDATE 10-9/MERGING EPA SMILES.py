import pandas as pd

master = pd.read_csv("AR_BLA_Master_Dataset.csv")
epa = pd.read_csv("SMILES.csv")

# Make column names consistent
master.columns = master.columns.str.strip().str.lower()
epa.columns = epa.columns.str.strip().str.lower()

# Extract DTXSID from master URLs
master["dtxsid_clean"] = (
    master["dtxsid"]
    .astype(str)
    .str.upper()
    .str.extract(r"(DTXSID\d+)", expand=False)
)

# Extract DTXSID from EPA file
epa["dtxsid_clean"] = (
    epa["dtxsid"]
    .astype(str)
    .str.upper()
    .str.extract(r"(DTXSID\d+)", expand=False)
)

print("MASTER examples:")
print(master["dtxsid_clean"].head())

print("\nEPA examples:")
print(epa["dtxsid_clean"].head())

# Keep EPA information needed
epa_small = epa[
    ["dtxsid_clean", "preferred_name", "smiles", "qsar_ready_smiles"]
].copy()

epa_small = epa_small.rename(columns={
    "preferred_name": "epa_name",
    "smiles": "epa_smiles",
    "qsar_ready_smiles": "epa_qsar_ready_smiles"
})

epa_small = epa_small.drop_duplicates("dtxsid_clean")

# Merge
merged = master.merge(
    epa_small,
    on="dtxsid_clean",
    how="left"
)

print("\nTotal rows:", len(merged))
print("Rows with EPA SMILES:", merged["epa_smiles"].notna().sum())
print("Rows missing EPA SMILES:", merged["epa_smiles"].isna().sum())

merged.to_csv("AR_BLA_Master_with_EPA_SMILES.csv", index=False)

print("\nSaved AR_BLA_Master_with_EPA_SMILES.csv")