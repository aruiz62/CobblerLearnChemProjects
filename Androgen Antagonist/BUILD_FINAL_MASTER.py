import pandas as pd
import re

# ============================================================
# FILE NAMES
# ============================================================

bla_primary_file = "AID_2283395_datatable_all.csv"
bla_viability_file = "AID_2283392_datatable_all.csv"

mdakb2_primary_file = "AID_2283747_datatable_all.csv"
mdakb2_viability_file = "AID_2283748_datatable_all.csv"


# ============================================================
# FUNCTION TO LOAD PUBCHEM FILE
# ============================================================

def load_pubchem(file, prefix):

    # Skip PubChem metadata rows 2-5
    df = pd.read_csv(
        file,
        skiprows=[1, 2, 3, 4],
        low_memory=False
    )

    print("\n======================================")
    print(prefix)
    print("======================================")
    print("Original rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # FIND DTXSID
    # --------------------------------------------------------

    # First check whether a DTXSID column already exists
    dtxsid_columns = [
        col for col in df.columns
        if "DTXSID" in str(col).upper()
    ]

    if dtxsid_columns:
        df["DTXSID"] = df[dtxsid_columns[0]].astype(str)

    else:
        # Search every column for DTXSID text
        def find_dtxsid(row):
            for value in row:
                if pd.notna(value):
                    match = re.search(
                        r"DTXSID\d+",
                        str(value),
                        flags=re.IGNORECASE
                    )

                    if match:
                        return match.group(0).upper()

            return None

        df["DTXSID"] = df.apply(find_dtxsid, axis=1)

    print("Rows with DTXSID:", df["DTXSID"].notna().sum())

    # Remove rows where DTXSID could not be identified
    df = df[df["DTXSID"].notna()].copy()

    # --------------------------------------------------------
    # CLEAN DTXSID
    # --------------------------------------------------------

    df["DTXSID"] = (
        df["DTXSID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # CHECK DUPLICATES
    # --------------------------------------------------------

    duplicate_count = df["DTXSID"].duplicated().sum()

    print("Duplicate DTXSID rows:", duplicate_count)

    # --------------------------------------------------------
    # COLLAPSE DUPLICATES
    #
    # We DO NOT blindly delete duplicates.
    # For each DTXSID, keep the first non-missing value
    # found for every column.
    # --------------------------------------------------------

    def first_valid(series):
        values = series.dropna()

        if len(values) == 0:
            return None

        return values.iloc[0]

    df = (
        df.groupby("DTXSID", as_index=False)
        .agg(first_valid)
    )

    print("Unique DTXSIDs after duplicate handling:", len(df))

    # --------------------------------------------------------
    # RENAME COLUMNS
    # --------------------------------------------------------

    rename_dictionary = {}

    for col in df.columns:

        if col != "DTXSID":
            rename_dictionary[col] = prefix + "_" + str(col)

    df = df.rename(columns=rename_dictionary)

    return df


# ============================================================
# LOAD ALL FOUR ASSAYS
# ============================================================

bla_primary = load_pubchem(
    bla_primary_file,
    "BLA_PRIMARY"
)

bla_viability = load_pubchem(
    bla_viability_file,
    "BLA_VIABILITY"
)

mdakb2_primary = load_pubchem(
    mdakb2_primary_file,
    "MDAKB2_PRIMARY"
)

mdakb2_viability = load_pubchem(
    mdakb2_viability_file,
    "MDAKB2_VIABILITY"
)


# ============================================================
# DATASET SIZES
# ============================================================

print("\n======================================")
print("UNIQUE CHEMICALS IN EACH DATASET")
print("======================================")

print("BLA primary:", len(bla_primary))
print("BLA viability:", len(bla_viability))
print("MDAKB2 primary:", len(mdakb2_primary))
print("MDAKB2 viability:", len(mdakb2_viability))


# ============================================================
# MERGE ALL FOUR BY DTXSID
#
# OUTER JOIN keeps chemicals even if they were not tested
# in every assay.
# ============================================================

master = pd.merge(
    bla_primary,
    bla_viability,
    on="DTXSID",
    how="outer"
)

master = pd.merge(
    master,
    mdakb2_primary,
    on="DTXSID",
    how="outer"
)

master = pd.merge(
    master,
    mdakb2_viability,
    on="DTXSID",
    how="outer"
)


# ============================================================
# MASTER DATASET CHECK
# ============================================================

print("\n======================================")
print("MASTER DATASET")
print("======================================")

print("Total rows:", len(master))
print("Unique DTXSIDs:", master["DTXSID"].nunique())
print("Total columns:", len(master))

duplicate_master = master["DTXSID"].duplicated().sum()

print("Duplicate DTXSIDs:", duplicate_master)


# ============================================================
# MISSING DATA
# ============================================================

missing = pd.DataFrame({
    "Missing_Count": master.isna().sum(),
    "Missing_Percent":
        (master.isna().mean() * 100).round(2)
})

print("\n======================================")
print("MISSING DATA")
print("======================================")

print(missing.to_string())


# ============================================================
# ACTIVITY COUNTS
# ============================================================

print("\n======================================")
print("ACTIVITY OUTCOMES")
print("======================================")

activity_columns = [
    col for col in master.columns
    if "ACTIVITY_OUTCOME" in col.upper()
]

for col in activity_columns:

    print("\n", col)

    counts = master[col].value_counts(
        dropna=False
    )

    percentages = master[col].value_counts(
        dropna=False,
        normalize=True
    ) * 100

    print("Counts:")
    print(counts)

    print("\nPercentages:")
    print(percentages.round(2))


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

numeric_columns = master.select_dtypes(
    include="number"
).columns

print("\n======================================")
print("NUMERICAL SUMMARY")
print("======================================")

if len(numeric_columns) > 0:
    print(
        master[numeric_columns]
        .describe()
        .round(3)
        .to_string()
    )
else:
    print("No numerical columns detected.")


# ============================================================
# SAVE RESULTS
# ============================================================

master_file = "AR_ANTAGONIST_FINAL_MASTER.csv"
missing_file = "AR_FINAL_MISSING_DATA.csv"

master.to_csv(
    master_file,
    index=False
)

missing.to_csv(
    missing_file
)


print("\n======================================")
print("SAVED")
print("======================================")

print(master_file)
print(missing_file)

print("\nFIRST 5 ROWS:")
print(master.head())

print("\nDONE!")