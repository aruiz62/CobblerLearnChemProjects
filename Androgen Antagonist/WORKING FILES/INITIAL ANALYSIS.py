import pandas as pd

file = "AR_ANTAGONIST_MASTER_DATASET.csv"

df = pd.read_csv(file)

print("\n================================")
print("ANDROGEN ANTAGONIST ANALYSIS")
print("================================")

print("\nTotal chemicals:", df["DTXSID"].nunique())
print("Total rows:", len(df))


# ============================================================
# PRIMARY ASSAY ACTIVITY
# ============================================================

assays = {
    "BLA Antagonist":
        "BLA_HIT CALL",

    "MDAKB2 Antagonist":
        "MDAKB2_HIT CALL"
}


for assay_name, column in assays.items():

    print("\n--------------------------------")
    print(assay_name)
    print("--------------------------------")

    if column in df.columns:

        tested = df[column].dropna()

        print("Chemicals with results:", len(tested))

        counts = tested.value_counts()

        print("\nHit call counts:")
        print(counts)

        percentages = (
            tested.value_counts(normalize=True) * 100
        ).round(2)

        print("\nPercentages:")
        print(percentages)

    else:
        print("Column not found:", column)


# ============================================================
# VIABILITY QA
# ============================================================

print("\n================================")
print("VIABILITY QA")
print("================================")

for column in [
    "BLA_VIABILITY_STATUS",
    "MDAKB2_VIABILITY_STATUS"
]:

    print("\n", column)

    if column in df.columns:

        print(df[column].value_counts(dropna=False))

        print("\nPercent:")
        print(
            (
                df[column]
                .value_counts(
                    normalize=True,
                    dropna=False
                ) * 100
            ).round(2)
        )


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

print("\n================================")
print("NUMERICAL VARIABLES")
print("================================")

numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    df[numeric_columns]
    .describe()
    .round(3)
    .to_string()
)


# ============================================================
# MISSING DATA
# ============================================================

print("\n================================")
print("MISSING DATA")
print("================================")

missing = pd.DataFrame({
    "Missing_Count":
        df.isna().sum(),

    "Missing_Percent":
        (df.isna().mean() * 100).round(2)
})

print(missing.to_string())


# Save missing data table

missing.to_csv(
    "AR_MISSING_DATA_SUMMARY.csv"
)

print("\nSaved:")
print("AR_MISSING_DATA_SUMMARY.csv")

print("\nANALYSIS COMPLETE!")