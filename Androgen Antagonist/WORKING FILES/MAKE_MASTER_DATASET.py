import pandas as pd
import re

# --------------------------------------------------
# FILES
# --------------------------------------------------

files = {
    "BLA_PRIMARY": "AR BLA ANTAGONIST RATIO.csv",
    "BLA_VIABILITY": "AR BLA ANTAGONIST VIABILITY.csv",
    "MDAKB2_PRIMARY": "AR LUC MDA-KB2 ANTAGONIST.csv",
    "MDAKB2_VIABILITY": "MDA-KB2 ANTAGONIST VIABILITY.csv"
}


# --------------------------------------------------
# FUNCTION TO LOAD EACH PUBCHEM FILE
# --------------------------------------------------

def load_assay(filename, assay_name):

    df = pd.read_csv(
        filename,
        skiprows=[1, 2, 3, 4],
        low_memory=False
    )

    # Extract DTXSID from PubChem activity URL
    df["DTXSID"] = (
        df["PUBCHEM_ACTIVITY_URL"]
        .astype(str)
        .str.extract(r"(DTXSID\d+)", expand=False)
    )

    # Keep only rows where a DTXSID was found
    df = df[df["DTXSID"].notna()].copy()

    # Keep useful columns
    keep = [
        "DTXSID",
        "PUBCHEM_SID",
        "PUBCHEM_CID",
        "PUBCHEM_EXT_DATASOURCE_SMILES",
        "PUBCHEM_ACTIVITY_OUTCOME",
        "PUBCHEM_ACTIVITY_SCORE",
        "AC50",
        "HITC",
        "BMD"
    ]

    keep = [col for col in keep if col in df.columns]
    df = df[keep].copy()

    # Rename columns so we know which assay they came from
    rename = {}

    for col in df.columns:
        if col != "DTXSID":
            rename[col] = assay_name + "_" + col

    df = df.rename(columns=rename)

    # One row per DTXSID
    # Keeps the first PubChem record for duplicated DTXSIDs
    before = len(df)

    df = df.drop_duplicates(
        subset="DTXSID",
        keep="first"
    )

    print("\n" + assay_name)
    print("-" * 50)
    print("Rows before duplicate handling:", before)
    print("Unique DTXSIDs:", len(df))

    return df


# --------------------------------------------------
# LOAD FOUR ASSAYS
# --------------------------------------------------

bla_primary = load_assay(
    files["BLA_PRIMARY"],
    "BLA_PRIMARY"
)

bla_viability = load_assay(
    files["BLA_VIABILITY"],
    "BLA_VIABILITY"
)

mdakb2_primary = load_assay(
    files["MDAKB2_PRIMARY"],
    "MDAKB2_PRIMARY"
)

mdakb2_viability = load_assay(
    files["MDAKB2_VIABILITY"],
    "MDAKB2_VIABILITY"
)


# --------------------------------------------------
# MERGE USING DTXSID
# --------------------------------------------------

master = bla_primary.merge(
    bla_viability,
    on="DTXSID",
    how="outer"
)

master = master.merge(
    mdakb2_primary,
    on="DTXSID",
    how="outer"
)

master = master.merge(
    mdakb2_viability,
    on="DTXSID",
    how="outer"
)


# --------------------------------------------------
# CHECK MASTER DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("MASTER DATASET")
print("=" * 60)

print("Total unique chemicals:", master["DTXSID"].nunique())
print("Total rows:", len(master))
print("Total columns:", len(master.columns))

print("\nDuplicate DTXSIDs in master:")
print(master["DTXSID"].duplicated().sum())


# --------------------------------------------------
# SAVE MASTER DATASET
# --------------------------------------------------

master.to_csv(
    "AR_ANTAGONIST_MASTER_DATASET_NEW.csv",
    index=False
)

print("\nSaved:")
print("AR_ANTAGONIST_MASTER_DATASET_NEW.csv")

print("\nFIRST 5 ROWS:")
print(master.head())

print("\nDONE!")