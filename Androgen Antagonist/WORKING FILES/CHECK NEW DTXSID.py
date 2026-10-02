import pandas as pd
import re

# =========================================================
# CHECK AID 2283392 FOR DTXSIDs
# =========================================================

file = "../AID_2283392_datatable_all.csv"

# Skip PubChem metadata rows
df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

print("=" * 50)
print("COLUMN NAMES")
print("=" * 50)

for i, col in enumerate(df.columns):
    print(i, col)

# =========================================================
# FIND DTXSID IN ALL COLUMNS
# =========================================================

def find_dtxsid(row):
    for value in row:
        if pd.notna(value):
            match = re.search(r"DTXSID\d+", str(value))
            if match:
                return match.group(0)
    return None


df["FOUND_DTXSID"] = df.apply(find_dtxsid, axis=1)

# =========================================================
# SHOW WHERE DTXSID APPEARS
# =========================================================

print("\n" + "=" * 50)
print("DTXSID LOCATIONS")
print("=" * 50)

for col in df.columns:
    count = df[col].astype(str).str.contains(
        r"DTXSID\d+",
        regex=True,
        na=False
    ).sum()

    if count > 0:
        print(f"\nCOLUMN: {col}")
        print(f"Rows containing DTXSID: {count}")

# =========================================================
# SUMMARY
# =========================================================

total_rows = len(df)
rows_with_dtxsid = df["FOUND_DTXSID"].notna().sum()
unique_dtxsids = df["FOUND_DTXSID"].nunique()
duplicate_rows = df["FOUND_DTXSID"].duplicated(keep=False).sum()

print("\n" + "=" * 50)
print("DTXSID SUMMARY")
print("=" * 50)

print("Total rows:", total_rows)
print("Rows with DTXSID:", rows_with_dtxsid)
print("Unique DTXSIDs:", unique_dtxsids)
print("Duplicate DTXSID rows:", duplicate_rows)

print("\nFirst 10 DTXSIDs:")
print(df["FOUND_DTXSID"].dropna().head(10).to_string(index=False))

# =========================================================
# SHOW EXAMPLE DUPLICATES
# =========================================================

duplicates = df[
    df["FOUND_DTXSID"].notna()
    & df["FOUND_DTXSID"].duplicated(keep=False)
].copy()

print("\n" + "=" * 50)
print("EXAMPLE DUPLICATE RECORDS")
print("=" * 50)

show_cols = ["FOUND_DTXSID"]

for col in [
    "PUBCHEM_SID",
    "PUBCHEM_CID",
    "PUBCHEM_ACTIVITY_OUTCOME"
]:
    if col in df.columns:
        show_cols.append(col)

print(duplicates[show_cols].head(30).to_string(index=False))

print("\nDONE!")