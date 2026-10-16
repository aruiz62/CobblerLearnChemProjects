
import pandas as pd
import requests
import time
from pathlib import Path
from urllib.parse import quote

# STEP 1: Load master dataset
folder = Path(__file__).resolve().parent

df = pd.read_csv(
    folder / "AR_BLA_Master_Dataset.csv",
    dtype=str
)

df["chemical_id"] = df["dtxsid"].str.extract(
    r"(DTXSID\d+)", expand=False
)

cache_file = folder / "PubChem_SMILES_Progress.csv"
output_file = folder / "AR_BLA_Master_with_SMILES.csv"

# None = process all remaining chemicals
MAX_NEW_LOOKUPS = None

# STEP 2: Restore saved progress
if cache_file.exists():
    saved = pd.read_csv(cache_file, dtype=str)
    saved = saved.drop_duplicates(
        subset="chemical_id", keep="last"
    )
    records = saved.set_index("chemical_id").to_dict("index")
else:
    records = {}

completed_statuses = {"Found", "Not found", "Missing CAS"}

completed = {
    chem_id for chem_id, record in records.items()
    if record.get("status") in completed_statuses
}

print("Total chemicals:", len(df))
print("Previously completed:", len(completed))

# STEP 3: Search PubChem with retries
session = requests.Session()

def get_smiles(cas):
    encoded_cas = quote(str(cas).strip(), safe="")

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{encoded_cas}/"
        "property/SMILES,ConnectivitySMILES/JSON"
    )

    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)

            if response.status_code == 404:
                return None, None, None, "Not found"

            if response.status_code == 400:
                return None, None, None, "Review identifier"

            if response.status_code in (403, 429, 500, 502, 503, 504):
                print(
                    f"HTTP {response.status_code} for CAS {cas}"
                )
                time.sleep(15 * (attempt + 1))
                continue

            response.raise_for_status()

            props = response.json()[
                "PropertyTable"
            ]["Properties"][0]

            return (
                props.get("CID"),
                props.get("SMILES"),
                props.get("ConnectivitySMILES"),
                "Found"
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            IndexError
        ) as error:
            print(f"Attempt {attempt + 1}: {error}")
            time.sleep(10 * (attempt + 1))

    return None, None, None, "Retry later"


# STEP 4: Save progress function
def save_progress():
    progress = pd.DataFrame([
        {"chemical_id": key, **value}
        for key, value in records.items()
    ])

    progress.to_csv(cache_file, index=False)

    combined = df.merge(
        progress,
        on="chemical_id",
        how="left",
        validate="one_to_one"
    )

    combined.to_csv(output_file, index=False)
    return combined


# STEP 5: Retrieve remaining chemicals
new_lookups = 0
consecutive_errors = 0

try:
    for _, row in df.iterrows():
        chemical_id = row["chemical_id"]

        if pd.isna(chemical_id):
            continue

        if chemical_id in completed:
            continue

        cas = row["casrn_antagonist"]

        if pd.isna(cas) or not str(cas).strip():
            cid, smiles, connectivity, status = (
                None, None, None, "Missing CAS"
            )
        else:
            cid, smiles, connectivity, status = get_smiles(
                cas
            )

            # Slow requests to reduce server overload
            time.sleep(1)

        records[chemical_id] = {
            "pubchem_cid": cid,
            "SMILES": smiles,
            "ConnectivitySMILES": connectivity,
            "status": status
        }

        if status in completed_statuses:
            completed.add(chemical_id)

        new_lookups += 1

        if status == "Retry later":
            consecutive_errors += 1
            print(f"Temporary failure: {chemical_id}")
        else:
            consecutive_errors = 0

        if new_lookups % 10 == 0:
            save_progress()
            print("New chemicals checked:", new_lookups)

        # Stop if PubChem repeatedly fails
        if consecutive_errors >= 3:
            print("\nPubChem may be temporarily unavailable.")
            print("Progress saved. Try again later.")
            break

        if (MAX_NEW_LOOKUPS is not None and
                new_lookups >= MAX_NEW_LOOKUPS):
            break

except KeyboardInterrupt:
    print("\nStopped by user. Saving progress.")

finally:
    combined = save_progress()
    session.close()

print("\nFINAL RESULTS")
print("Total chemicals:", len(combined))
print("SMILES found:", combined["SMILES"].notna().sum())
print("SMILES not available:",
      combined["SMILES"].isna().sum())

print("\nLOOKUP STATUS:")
print(combined["status"].value_counts(dropna=False))

print("\nProgress saved successfully!")
