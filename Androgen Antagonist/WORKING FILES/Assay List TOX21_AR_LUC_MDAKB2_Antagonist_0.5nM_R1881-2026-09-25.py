import pandas as pd

file = "/Androgen Antagonist/WORKING FILES/Assay List TOX21_AR_BLA_Antagonist_ratio-2026-09-25.csv"

df = pd.read_csv(file)

print("NUMBER OF ROWS:", len(df))

print("\nALL COLUMNS:")
for col in df.columns:
    print(col)
