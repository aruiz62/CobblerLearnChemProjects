import pandas as pd
import glob
import os

# Find the two ORIGINAL primary assay CSV files
files = glob.glob("*.csv")

bla_file = None
mdakb2_file = None

for file in files:
    name = os.path.basename(file).upper()

    # Ignore viability and master files
    if "VIABILITY" in name or "MASTER" in name:
        continue

    if "BLA_ANTAGONIST_RATIO" in name:
        bla_file = file

    if "MDAKB2_ANTAGONIST_0.5NM_R1881" in name:
        mdakb2_file = file


def check_assay(file, assay_name):

    print("\n" + "=" * 60)
    print(assay_name)
    print("=" * 60)

    if file is None:
        print("FILE NOT FOUND")
        return

    print("File:", file)

    df = pd.read_csv(file)

    print("\nTotal rows:", len(df))
    print("Unique DTXSIDs:", df["DTXSID"].nunique())

    print("\nHIT CALL:")
    if "HIT CALL" in df.columns:
        print(df["HIT CALL"].value_counts(dropna=False))
    else:
        print("No HIT CALL column")

    print("\nCONTINUOUS HIT CALL:")
    if "CONTINUOUS HIT CALL" in df.columns:
        print(df["CONTINUOUS HIT CALL"].value_counts(dropna=False))
    else:
        print("No CONTINUOUS HIT CALL column")

    print("\nTOXCAST ACTIVE:")
    if "TOXCAST ACTIVE" in df.columns:
        print(df["TOXCAST ACTIVE"].value_counts(dropna=False))
    else:
        print("No TOXCAST ACTIVE column")

    print("\nFirst 10 HIT CALL values:")
    if "HIT CALL" in df.columns:
        print(df["HIT CALL"].head(10))


check_assay(
    bla_file,
    "TOX21 AR BLA ANTAGONIST"
)

check_assay(
    mdakb2_file,
    "TOX21 AR MDAKB2 ANTAGONIST"
)