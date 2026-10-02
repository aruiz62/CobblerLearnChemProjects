import pandas as pd
from rdkit import Chem

file = "AR BLA ANTAGONIST VIABILITY.csv"

df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

def smiles_to_inchikey(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return Chem.MolToInchiKey(mol)
    except:
        pass
    return None

df["InChIKey"] = df["PUBCHEM_EXT_DATASOURCE_SMILES"].apply(smiles_to_inchikey)

# Keep identifiers we need
output = df[
    ["PUBCHEM_CID", "PUBCHEM_EXT_DATASOURCE_SMILES", "InChIKey"]
].drop_duplicates()

output.to_csv("CID_InChIKey.csv", index=False)

print("Total rows:", len(output))
print("InChIKeys created:", output["InChIKey"].notna().sum())
print("Missing InChIKeys:", output["InChIKey"].isna().sum())

print("\nFIRST 10:")
print(output.head(10))

print("\nSaved: CID_InChIKey.csv")

# Create a text file containing only the InChIKeys
output["InChIKey"].dropna().to_csv(
    "InChIKeys.txt",
    index=False,
    header=False
)

print("Saved: InChIKeys.txt")