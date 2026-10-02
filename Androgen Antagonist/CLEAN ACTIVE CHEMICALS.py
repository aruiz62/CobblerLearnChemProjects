import pandas as pd

# Load the comparison file you just created
df = pd.read_csv("AR_ASSAY_COMPARISON.csv")

# Keep ONLY chemicals that:
# 1. Were active in BLA
# 2. Were active in MDA-kb2
# 3. Did NOT show BLA viability activity
# 4. Did NOT show MDA-kb2 viability activity

clean_active = df[
    (df["BLA_PRIMARY"].str.lower() == "active") &
    (df["MDAKB2_PRIMARY"].str.lower() == "active") &
    (df["BLA_VIABILITY"].str.lower() == "inactive") &
    (df["MDAKB2_VIABILITY"].str.lower() == "inactive")
].copy()

# Add an easier interpretation column
clean_active["RESULT"] = (
    "Active in both androgen assays; "
    "no viability activity detected"
)

# Show table
print("\n==============================================")
print("POTENTIAL ANDROGEN ANTAGONISTS")
print("ACTIVE IN BOTH ASSAYS + VIABILITY INACTIVE")
print("==============================================\n")

print(clean_active.to_string(index=False))

print("\nTotal chemicals:", len(clean_active))

# Save as a new CSV
clean_active.to_csv(
    "POTENTIAL_ANDROGEN_ANTAGONISTS.csv",
    index=False
)

print("\nSaved as POTENTIAL_ANDROGEN_ANTAGONISTS.csv")