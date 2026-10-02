import pandas as pd

# Load BLA viability dataset
file = "AR BLA ANTAGONIST VIABILITY.csv"

df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

# Extract DTXSID directly from the CompTox URL
df["DTXSID"] = df["PUBCHEM_ACTIVITY_URL"].str.extract(
    r"(DTXSID\d+)",
    expand=False
)

# Check results
print("TOTAL ROWS:", len(df))
print("DTXSIDs FOUND:", df["DTXSID"].notna().sum())
print("MISSING DTXSIDs:", df["DTXSID"].isna().sum())

print("\nFIRST 10:")
print(
    df[
        ["DTXSID", "PUBCHEM_CID",
         "PUBCHEM_ACTIVITY_OUTCOME", "AC50", "HITC", "BMD"]
    ].head(10)
)

# Save cleaned viability dataset
df.to_csv("BLA_VIABILITY_WITH_DTXSID.csv", index=False)

print("\nSaved: BLA_VIABILITY_WITH_DTXSID.csv")