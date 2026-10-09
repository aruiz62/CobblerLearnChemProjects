
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

df = pd.read_csv(
    folder / "AR_BLA_Master_Dataset.csv"
)

columns = [
    "preferred_name_antagonist",
    "hit_call_antagonist",
    "ac50_antagonist",
    "hit_call_viability",
    "ac50_viability"
]

# Show the first 15 chemicals
print("\nFIRST 15 CHEMICALS:")
print(df[columns].head(15).to_string(index=False))

# Check AC50 values for active chemicals
active = df[
    df["hit_call_antagonist"] == "Active"
]

print("\nACTIVE ANTAGONIST AC50 VALUES:")
print(active["ac50_antagonist"].head(20).to_string(index=False))

# Check the types of values stored
for col in ["ac50_antagonist", "ac50_viability"]:
    numeric = pd.to_numeric(df[col], errors="coerce")

    print("\n", col)
    print("Numeric values:", numeric.notna().sum())
    print("Non-numeric values:", numeric.isna().sum())
    print("Minimum:", numeric.min())
    print("Maximum:", numeric.max())
