import pandas as pd
from pathlib import Path

# =========================================================
# 1. FIND AND LOAD MASTER DATASET
# =========================================================

folder = Path(__file__).parent

possible_files = [
    folder / "AR_ANTAGONIST_MASTER_DATASET_NEW.csv",
    folder / "AR_ANTAGONIST_MASTER_DATASET.csv",
    folder.parent / "AR_ANTAGONIST_MASTER_DATASET_NEW.csv",
    folder.parent / "AR_ANTAGONIST_MASTER_DATASET.csv"
]

master_file = None

for file in possible_files:
    if file.exists():
        master_file = file
        break

if master_file is None:
    raise FileNotFoundError(
        "Could not find AR_ANTAGONIST_MASTER_DATASET_NEW.csv "
        "or AR_ANTAGONIST_MASTER_DATASET.csv"
    )

df = pd.read_csv(master_file)

print("\nMASTER DATASET LOADED:")
print(master_file.name)


# =========================================================
# 2. FUNCTION TO FIND COLUMNS AUTOMATICALLY
# =========================================================

def find_column(required_words):
    for column in df.columns:

        column_upper = column.upper()

        if all(word.upper() in column_upper for word in required_words):
            return column

    return None


dtxsid_col = find_column(["DTXSID"])

bla_primary_col = find_column(
    ["BLA", "PRIMARY", "ACTIVITY", "OUTCOME"]
)

bla_viability_col = find_column(
    ["BLA", "VIABILITY", "ACTIVITY", "OUTCOME"]
)

mda_primary_col = find_column(
    ["MDA", "PRIMARY", "ACTIVITY", "OUTCOME"]
)

mda_viability_col = find_column(
    ["MDA", "VIABILITY", "ACTIVITY", "OUTCOME"]
)


# =========================================================
# 3. SHOW WHICH COLUMNS WERE FOUND
# =========================================================

print("\nCOLUMNS FOUND:")

print("DTXSID:", dtxsid_col)
print("BLA Primary:", bla_primary_col)
print("BLA Viability:", bla_viability_col)
print("MDA-kb2 Primary:", mda_primary_col)
print("MDA-kb2 Viability:", mda_viability_col)


# Make sure all necessary columns were found
needed_columns = [
    dtxsid_col,
    bla_primary_col,
    bla_viability_col,
    mda_primary_col,
    mda_viability_col
]

if any(column is None for column in needed_columns):

    print("\nCould not automatically find every column.")
    print("\nALL COLUMNS IN YOUR DATASET:")

    for column in df.columns:
        print(column)

    raise ValueError(
        "One or more required columns could not be found."
    )


# =========================================================
# 4. CREATE COMPARISON TABLE
# =========================================================

comparison = df[
    [
        dtxsid_col,
        bla_primary_col,
        bla_viability_col,
        mda_primary_col,
        mda_viability_col
    ]
].copy()


# Rename columns so table is easier to read
comparison.columns = [
    "DTXSID",
    "BLA_PRIMARY",
    "BLA_VIABILITY",
    "MDAKB2_PRIMARY",
    "MDAKB2_VIABILITY"
]


# =========================================================
# 5. INTERPRET EACH CHEMICAL
# =========================================================

def clean_result(value):

    if pd.isna(value):
        return "missing"

    return str(value).strip().lower()


def interpret(row):

    bla = clean_result(row["BLA_PRIMARY"])
    bla_v = clean_result(row["BLA_VIABILITY"])

    mda = clean_result(row["MDAKB2_PRIMARY"])
    mda_v = clean_result(row["MDAKB2_VIABILITY"])

    # Both primary assays active and viability assays inactive
    if (
        bla == "active"
        and mda == "active"
        and bla_v == "inactive"
        and mda_v == "inactive"
    ):
        return "Both primary assays active - no viability interference detected"

    # BLA primary active AND its viability assay active
    elif (
        bla == "active"
        and bla_v == "active"
    ):
        return "Possible BLA viability interference"

    # MDA primary active AND its viability assay active
    elif (
        mda == "active"
        and mda_v == "active"
    ):
        return "Possible MDA-kb2 viability interference"

    # Primary assays disagree
    elif (
        bla == "active"
        and mda == "inactive"
    ):
        return "BLA active only"

    elif (
        bla == "inactive"
        and mda == "active"
    ):
        return "MDA-kb2 active only"

    # Neither primary assay active
    elif (
        bla == "inactive"
        and mda == "inactive"
    ):
        return "Inactive in both primary assays"

    else:
        return "Missing or unavailable data"


comparison["INTERPRETATION"] = comparison.apply(
    interpret,
    axis=1
)


# =========================================================
# 6. PRINT RESULTS
# =========================================================

print("\n============================================")
print("ASSAY COMPARISON RESULTS")
print("============================================")

print(comparison.head(30))


# =========================================================
# 7. COUNT EACH TYPE OF RESULT
# =========================================================

print("\n============================================")
print("INTERPRETATION COUNTS")
print("============================================")

print(comparison["INTERPRETATION"].value_counts())


# =========================================================
# 8. SHOW CHEMICALS ACTIVE IN BOTH PRIMARY ASSAYS
# =========================================================

both_active = comparison[
    (comparison["BLA_PRIMARY"].astype(str).str.lower() == "active")
    &
    (comparison["MDAKB2_PRIMARY"].astype(str).str.lower() == "active")
]

print("\n============================================")
print("CHEMICALS ACTIVE IN BOTH PRIMARY ASSAYS")
print("============================================")

print(both_active)


# =========================================================
# 9. SHOW POSSIBLE VIABILITY INTERFERENCE
# =========================================================

possible_interference = comparison[
    comparison["INTERPRETATION"].str.contains(
        "interference",
        case=False,
        na=False
    )
]

print("\n============================================")
print("POSSIBLE VIABILITY INTERFERENCE")
print("============================================")

print(possible_interference)


# =========================================================
# 10. SAVE RESULTS
# =========================================================

output_file = folder / "AR_ASSAY_COMPARISON.csv"

comparison.to_csv(output_file, index=False)

print("\n============================================")
print("DONE")
print("============================================")

print("Saved as:")
print(output_file.name)