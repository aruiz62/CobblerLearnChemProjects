import pandas as pd

# Load formatted viability datasets
bla = pd.read_csv("BLA_VIABILITY_WITH_DTXSID.csv")
mda = pd.read_csv("MDAKB2_VIABILITY_WITH_DTXSID.csv")

print("\n===== BLA DUPLICATE EXAMPLE =====")

bla_duplicates = bla[
    bla["DTXSID"].duplicated(keep=False)
].sort_values("DTXSID")

print(bla_duplicates.head(20).to_string(index=False))


print("\n===== MDAKB2 DUPLICATE EXAMPLE =====")

mda_duplicates = mda[
    mda["DTXSID"].duplicated(keep=False)
].sort_values("DTXSID")

print(mda_duplicates.head(20).to_string(index=False))