import pandas as pd

# --------------------------------------------------
# LOAD FINAL MASTER DATASET
# --------------------------------------------------

file = "AR_ANTAGONIST_MASTER_DATASET_NEW.csv"
df = pd.read_csv(file)

print("=" * 80)
print("ANDROGEN ANTAGONIST DATA DICTIONARY")
print("=" * 80)

print("\nColumns in master dataset:", len(df.columns))


# --------------------------------------------------
# FUNCTION TO IDENTIFY ASSAY
# --------------------------------------------------

def get_assay(column):

    if column == "DTXSID":
        return "Chemical Identifier"

    elif column.startswith("BLA_PRIMARY"):
        return "TOX21_AR_BLA_Antagonist_ratio"

    elif column.startswith("BLA_VIABILITY"):
        return "TOX21_AR_BLA_Antagonist_viability"

    elif column.startswith("MDAKB2_PRIMARY"):
        return "TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881"

    elif column.startswith("MDAKB2_VIABILITY"):
        return "TOX21_AR_LUC_MDAKB2_Antagonist_0.5nM_R1881_viability"

    else:
        return "General"


# --------------------------------------------------
# FUNCTION TO DESCRIBE EACH COLUMN
# --------------------------------------------------

def describe_column(column):

    # Chemical identifier
    if column == "DTXSID":
        return (
            "EPA DSSTox Substance Identifier used to match the same chemical "
            "across assay datasets.",
            "Identifier"
        )

    # PubChem identifiers
    elif "PUBCHEM_RESULT_TAG" in column:
        return (
            "PubChem row/result identifier for the assay record.",
            "Identifier"
        )

    elif "PUBCHEM_SID" in column:
        return (
            "PubChem Substance Identifier (SID) for the tested substance.",
            "Identifier"
        )

    elif "PUBCHEM_CID" in column:
        return (
            "PubChem Compound Identifier (CID) associated with the chemical.",
            "Identifier"
        )

    # Structure
    elif "PUBCHEM_EXT_DATASOURCE_SMILES" in column:
        return (
            "SMILES representation of the chemical structure supplied in the PubChem assay data.",
            "Structure string"
        )

    # Activity classification
    elif "PUBCHEM_ACTIVITY_OUTCOME" in column:
        if "VIABILITY" in column:
            return (
                "PubChem activity classification for the viability assay. "
                "Used as quality-assessment information rather than receptor activity.",
                "Categorical"
            )
        else:
            return (
                "PubChem activity classification for the primary androgen receptor antagonist assay.",
                "Categorical"
            )

    elif "PUBCHEM_ACTIVITY_SCORE" in column:
        return (
            "PubChem activity score associated with the assay result.",
            "Unitless score"
        )

    elif "PUBCHEM_ACTIVITY_URL" in column:
        return (
            "URL linking to additional assay or chemical activity information.",
            "URL"
        )

    elif "PUBCHEM_ASSAYDATA_COMMENT" in column:
        return (
            "Comment or annotation supplied with the PubChem assay result.",
            "Text"
        )

    # Numerical assay measurements
    elif column.endswith("_AC50"):
        return (
            "Concentration producing 50% of the measured assay response.",
            "µM"
        )

    elif column.endswith("_HITC"):
        return (
            "Hit-call value associated with the assay activity classification.",
            "Unitless"
        )

    elif column.endswith("_BMD"):
        return (
            "Benchmark dose or concentration value reported for the assay.",
            "µM"
        )

    # Safety fallback so EVERY column gets defined
    else:
        return (
            "Variable retained from the original assay dataset.",
            "See source data"
        )


# --------------------------------------------------
# BUILD DATA DICTIONARY
# --------------------------------------------------

rows = []

for column in df.columns:

    description, units = describe_column(column)

    rows.append({
        "Variable": column,
        "Assay": get_assay(column),
        "Description": description,
        "Units": units
    })


data_dictionary = pd.DataFrame(rows)


# --------------------------------------------------
# DISPLAY AND SAVE
# --------------------------------------------------

print("\n" + "=" * 80)
print(data_dictionary.to_string(index=False))
print("=" * 80)

print("\nMaster dataset columns:", len(df.columns))
print("Dictionary rows:", len(data_dictionary))

if len(df.columns) == len(data_dictionary):
    print("\nSUCCESS: Every master-dataset column is included in the dictionary.")
else:
    print("\nWARNING: Dictionary does not contain every column.")


data_dictionary.to_csv(
    "AR_DATA_DICTIONARY.csv",
    index=False
)

print("\nSaved: AR_DATA_DICTIONARY.csv")
print("\nDONE!")