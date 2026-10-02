import pandas as pd
import re

files = [
    "AR BLA ANTAGONIST VIABILITY.csv",
    "AR BLA ANTAGONIST RATIO.csv",
    "AR LUC MDA-KB2 ANTAGONIST.csv",
    "MDA-KB2 ANTAGONIST VIABILITY.csv"
]

for file in files:

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

    # Extract DTXSID
    df["DTXSID"] = (
        df["PUBCHEM_ACTIVITY_URL"]
        .astype(str)
        .str.extract(r"(DTXSID\d+)", expand=False)
    )

    print("Total rows:", len(df))
    print("Rows with DTXSID:", df["DTXSID"].notna().sum())
    print("Unique DTXSIDs:", df["DTXSID"].nunique())

    # Activity results
    if "PUBCHEM_ACTIVITY_OUTCOME" in df.columns:
        print("\nActivity outcomes:")
        print(df["PUBCHEM_ACTIVITY_OUTCOME"].value_counts(dropna=False))

    # Check numerical assay columns
    print("\nAssay result columns:")
    for col in ["AC50", "HITC", "BMD"]:
        if col in df.columns:
            print(col)

print("\n" + "=" * 70)
print("DONE CHECKING ALL FOUR FILES")
print("=" * 70)
