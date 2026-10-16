
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

# Load downloaded chemical structures
df = pd.read_csv(
    folder / "Chemical_Structures.csv",
    dtype=str
)

print("TOTAL ROWS:", len(df))

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Check missing data
print("\nMISSING VALUES:")
print(df.isna().sum().to_string())

# Display first 5 chemicals
print("\nFIRST 5 CHEMICALS:")
print(df.head(5).to_string(index=False))

print("\nFinished checking structures!")
