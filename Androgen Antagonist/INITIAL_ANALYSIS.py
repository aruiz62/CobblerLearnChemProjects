import pandas as pd

# Load master dataset
df = pd.read_csv("AR_ANTAGONIST_MASTER_DATASET_NEW.csv", low_memory=False)

print("=" * 60)
print("ANDROGEN ANTAGONIST - INITIAL ANALYSIS")
print("=" * 60)

print("\nTotal chemicals:", len(df))
print("Unique DTXSIDs:", df["DTXSID"].nunique())


# --------------------------------------------------
# PRIMARY ASSAY 1: BLA ANTAGONIST
# --------------------------------------------------

col1 = "BLA_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME"

print("\n" + "=" * 60)
print("BLA ANTAGONIST PRIMARY ASSAY")
print("=" * 60)

counts1 = df[col1].value_counts(dropna=False)
print("\nCounts:")
print(counts1)

tested1 = df[col1].notna().sum()

print("\nPercentages among tested chemicals:")

for outcome, count in df[col1].dropna().value_counts().items():
    percent = (count / tested1) * 100
    print(f"{outcome}: {count} ({percent:.2f}%)")

print("Untested/unavailable:", df[col1].isna().sum())


# --------------------------------------------------
# PRIMARY ASSAY 2: MDA-KB2 ANTAGONIST
# --------------------------------------------------

col2 = "MDAKB2_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME"

print("\n" + "=" * 60)
print("MDA-KB2 ANTAGONIST PRIMARY ASSAY")
print("=" * 60)

counts2 = df[col2].value_counts(dropna=False)
print("\nCounts:")
print(counts2)

tested2 = df[col2].notna().sum()

print("\nPercentages among tested chemicals:")

for outcome, count in df[col2].dropna().value_counts().items():
    percent = (count / tested2) * 100
    print(f"{outcome}: {count} ({percent:.2f}%)")

print("Untested/unavailable:", df[col2].isna().sum())


# --------------------------------------------------
# NUMERICAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

numeric_cols = [
    col for col in df.columns
    if "AC50" in col or "HITC" in col or "BMD" in col
]

print(df[numeric_cols].describe())


print("\nDONE!")