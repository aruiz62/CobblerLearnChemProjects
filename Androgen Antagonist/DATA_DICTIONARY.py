import pandas as pd

# Brief data dictionary for Androgen Antagonist project
data_dictionary = pd.DataFrame({
    "Variable": [
        "DTXSID",
        "BLA_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME",
        "BLA_PRIMARY_AC50",
        "BLA_PRIMARY_HITC",
        "BLA_PRIMARY_BMD",
        "BLA_VIABILITY_PUBCHEM_ACTIVITY_OUTCOME",
        "BLA_VIABILITY_AC50",
        "MDAKB2_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME",
        "MDAKB2_PRIMARY_AC50",
        "MDAKB2_PRIMARY_HITC",
        "MDAKB2_PRIMARY_BMD",
        "MDAKB2_VIABILITY_PUBCHEM_ACTIVITY_OUTCOME",
        "MDAKB2_VIABILITY_AC50"
    ],

    "Description": [
        "EPA DSSTox Substance Identifier used to match chemicals across assays.",
        "Activity classification for the AR BLA antagonist primary assay.",
        "Concentration producing 50% of the assay response in the BLA primary assay.",
        "Hit-call value associated with the BLA primary assay.",
        "Benchmark dose/concentration value reported for the BLA primary assay.",
        "Viability activity classification used as a QA check for the BLA assay.",
        "Concentration producing 50% response in the BLA viability assay.",
        "Activity classification for the AR MDA-KB2 antagonist primary assay.",
        "Concentration producing 50% of the assay response in the MDA-KB2 primary assay.",
        "Hit-call value associated with the MDA-KB2 primary assay.",
        "Benchmark dose/concentration value reported for the MDA-KB2 primary assay.",
        "Viability activity classification used as a QA check for the MDA-KB2 assay.",
        "Concentration producing 50% response in the MDA-KB2 viability assay."
    ],

    "Units": [
        "Identifier",
        "Categorical",
        "µM",
        "Unitless",
        "µM",
        "Categorical",
        "µM",
        "Categorical",
        "µM",
        "Unitless",
        "µM",
        "Categorical",
        "µM"
    ]
})

print("=" * 80)
print("ANDROGEN ANTAGONIST DATA DICTIONARY")
print("=" * 80)

print(data_dictionary.to_string(index=False))

# Save as CSV
data_dictionary.to_csv(
    "AR_DATA_DICTIONARY.csv",
    index=False
)

print("\nSaved: AR_DATA_DICTIONARY.csv")
print("\nDONE!")