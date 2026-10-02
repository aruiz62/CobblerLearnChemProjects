import pandas as pd

# Load master dataset
df = pd.read_csv(
    "AR_ANTAGONIST_MASTER_DATASET_NEW.csv",
    low_memory=False
)

print("=" * 70)
print("MISSING DATA ANALYSIS")
print("=" * 70)

# Count missing values
missing_count = df.isna().sum()

# Calculate percent missing
missing_percent = (missing_count / len(df)) * 100

# Create summary table
missing_summary = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percent": missing_percent
})

# Sort from most missing to least missing
missing_summary = missing_summary.sort_values(
    "Missing_Percent",
    ascending=False
)

print("\nMISSING VALUES BY COLUMN:\n")
print(missing_summary.to_string())

# Save results
missing_summary.to_csv(
    "AR_MISSING_DATA_SUMMARY_NEW.csv"
)

print("\n" + "=" * 70)
print("IMPORTANT: PRIMARY ASSAY COVERAGE")
print("=" * 70)

primary_columns = [
    "BLA_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME",
    "MDAKB2_PRIMARY_PUBCHEM_ACTIVITY_OUTCOME"
]

for col in primary_columns:

    tested = df[col].notna().sum()
    unavailable = df[col].isna().sum()

    print(f"\n{col}")
    print("Tested:", tested)
    print("Untested/unavailable:", unavailable)
    print(
        "Percent unavailable:",
        round((unavailable / len(df)) * 100, 2),
        "%"
    )

print("\nNOTE:")
print("Missing assay results are treated as untested/unavailable.")
print("They are NOT treated as inactive.")
print("No missing values were imputed.")

print("\nSaved: AR_MISSING_DATA_SUMMARY_NEW.csv")
print("\nDONE!")