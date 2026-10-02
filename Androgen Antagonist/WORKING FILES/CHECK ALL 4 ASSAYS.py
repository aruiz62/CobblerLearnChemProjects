import pandas as pd
import re

files = [
    "AID_2283392_datatable_all.csv",
    "AID_2283395_datatable_all.csv",
    "AID_2283747_datatable_all.csv",
    "AID_2283748_datatable_all.csv"
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
