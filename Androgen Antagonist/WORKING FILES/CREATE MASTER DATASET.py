import pandas as pd
import os

# ============================================================
# ANDROGEN ANTAGONIST MASTER DATASET
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# FIND PRIMARY ASSAY FILES
# ------------------------------------------------------------

bla_primary_file = None
mda_primary_file = None

for file in os.listdir(folder):

    if (
        file.endswith(".csv")
        and "TOX21_AR_BLA_Antagonist_ratio" in file
        and "VIABILITY" not in file.upper()
    ):
        bla_primary_file = os.path.join(folder, file)

    if (
        file.endswith(".csv")
        and "TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881" in file
        and "VIABILITY" not in file.upper()
    ):
        mda_primary_file = os.path.join(folder, file)


bla_viability_file = os.path.join(
    folder, "BLA_VIABILITY_WITH_DTXSID.csv"
)

mda_viability_file = os.path.join(
    folder, "MDAKB2_VIABILITY_WITH_DTXSID.csv"
)

if bla_primary_file is None:
    raise FileNotFoundError("BLA primary assay file not found.")

if mda_primary_file is None:
    raise FileNotFoundError("MDAKB2 primary assay file not found.")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

bla = pd.read_csv(bla_primary_file)
mda = pd.read_csv(mda_primary_file)
bla_v = pd.read_csv(bla_viability_file)
mda_v = pd.read_csv(mda_viability_file)

for df in [bla, mda, bla_v, mda_v]:
    df.columns = df.columns.str.strip()


print("\n==============================")
print("ORIGINAL DATA")
print("==============================")

print("BLA primary rows:", len(bla))
print("MDAKB2 primary rows:", len(mda))
print("BLA viability rows:", len(bla_v))
print("MDAKB2 viability rows:", len(mda_v))


# ------------------------------------------------------------
# CHECK PRIMARY DUPLICATES
# ------------------------------------------------------------

print("\n==============================")
print("PRIMARY ASSAY DUPLICATES")
print("==============================")

print("BLA duplicates:",
      bla["DTXSID"].duplicated().sum())

print("MDAKB2 duplicates:",
      mda["DTXSID"].duplicated().sum())


# ------------------------------------------------------------
# SUMMARIZE VIABILITY BY DTXSID
# ------------------------------------------------------------
# One DTXSID can have multiple PubChem viability records.
# We keep that information instead of randomly deleting rows.

def summarize_viability(df, prefix):

    # Find the activity outcome column
    activity_col = None

    for col in df.columns:
        if "ACTIVITY_OUTCOME" in col.upper():
            activity_col = col
            break

    if activity_col is None:
        raise ValueError(
            f"Could not find ACTIVITY_OUTCOME column for {prefix}"
        )

    temp = df.copy()

    # Convert activity outcome to text
    temp[activity_col] = (
        temp[activity_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Create active flag
    temp["VIABILITY_ACTIVE_FLAG"] = (
        temp[activity_col] == "active"
    ).astype(int)

    # Summarize all records belonging to each DTXSID
    summary = (
        temp.groupby("DTXSID", as_index=False)
        .agg(
            VIABILITY_RECORDS=("DTXSID", "size"),
            ACTIVE_VIABILITY_RECORDS=(
                "VIABILITY_ACTIVE_FLAG", "sum"
            ),
            ANY_VIABILITY_ACTIVE=(
                "VIABILITY_ACTIVE_FLAG", "max"
            )
        )
    )

    summary["VIABILITY_STATUS"] = summary[
        "ANY_VIABILITY_ACTIVE"
    ].map({
        1: "Active",
        0: "Inactive"
    })

    # Rename columns so BLA and MDAKB2 stay separate
    summary = summary.rename(columns={
        "VIABILITY_RECORDS":
            f"{prefix}_VIABILITY_RECORDS",

        "ACTIVE_VIABILITY_RECORDS":
            f"{prefix}_ACTIVE_VIABILITY_RECORDS",

        "ANY_VIABILITY_ACTIVE":
            f"{prefix}_ANY_VIABILITY_ACTIVE",

        "VIABILITY_STATUS":
            f"{prefix}_VIABILITY_STATUS"
    })

    return summary


bla_v_summary = summarize_viability(
    bla_v, "BLA"
)

mda_v_summary = summarize_viability(
    mda_v, "MDAKB2"
)


print("\n==============================")
print("VIABILITY AFTER SUMMARIZING")
print("==============================")

print(
    "Unique BLA viability DTXSIDs:",
    len(bla_v_summary)
)

print(
    "Unique MDAKB2 viability DTXSIDs:",
    len(mda_v_summary)
)


# ------------------------------------------------------------
# RENAME PRIMARY ASSAY COLUMNS
# ------------------------------------------------------------

bla = bla.rename(columns={
    col: "BLA_" + col
    for col in bla.columns
    if col != "DTXSID"
})

mda = mda.rename(columns={
    col: "MDAKB2_" + col
    for col in mda.columns
    if col != "DTXSID"
})


# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

master = pd.merge(
    bla,
    mda,
    on="DTXSID",
    how="outer"
)

master = pd.merge(
    master,
    bla_v_summary,
    on="DTXSID",
    how="outer"
)

master = pd.merge(
    master,
    mda_v_summary,
    on="DTXSID",
    how="outer"
)


# ------------------------------------------------------------
# FINAL CHECK
# ------------------------------------------------------------

print("\n==============================")
print("FINAL MASTER DATASET")
print("==============================")

print("Total rows:", len(master))

print(
    "Unique DTXSIDs:",
    master["DTXSID"].nunique()
)

print(
    "Duplicate DTXSIDs:",
    master["DTXSID"].duplicated().sum()
)

print(
    "Total columns:",
    len(master.columns)
)


# ------------------------------------------------------------
# MISSING DATA
# ------------------------------------------------------------

missing = pd.DataFrame({
    "Missing_Count":
        master.isna().sum(),

    "Missing_Percent":
        (master.isna().mean() * 100).round(2)
})

print("\n==============================")
print("MISSING DATA")
print("==============================")

print(missing.to_string())


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output_file = os.path.join(
    folder,
    "AR_ANTAGONIST_MASTER_DATASET.csv"
)

master.to_csv(
    output_file,
    index=False
)

print("\n==============================")
print("SAVED")
print("==============================")

print(output_file)

print("\nFIRST 5 ROWS:")
print(master.head())

print("\nDONE!")