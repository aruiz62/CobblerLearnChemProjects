import pandas as pd

# Load MDAKB2 viability dataset
file = "AR LUC MDAKB2 0.5 nM R1881 VIABILITY.csv"

df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

# Extract DTXSID from CompTox URL
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

# Save cleaned dataset
df.to_csv("MDAKB2_VIABILITY_WITH_DTXSID.csv", index=False)

print("\nSaved: MDAKB2_VIABILITY_WITH_DTXSID.csv")