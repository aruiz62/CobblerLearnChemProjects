
import pandas as pd
import urllib.request
import json
import re
import time
from pathlib import Path

folder = Path(__file__).resolve().parent / "NEW_MASTER"

master = pd.read_csv(
    folder / "AR_MASTER_PENDING_DTXSID.csv",
    dtype="string"
)

lookup_file = folder / "AR_DTXSID_LOOKUP_RESULTS.csv"

cids = (
    master["PUBCHEM_CID"]
    .dropna()
    .str.strip()
    .unique()
    .tolist()
)

# Load progress from an earlier run, if available
results = {}

if lookup_file.exists():
    old = pd.read_csv(lookup_file, dtype="string").fillna("")
    for row in old.to_dict("records"):
        results[row["PUBCHEM_CID"]] = row

def save_progress():
    columns = ["PUBCHEM_CID", "DTXSID", "CANDIDATES", "STATUS"]

    rows = [
        results[cid]
        for cid in cids
        if cid in results
    ]

    pd.DataFrame(rows, columns=columns).to_csv(
        lookup_file, index=False
    )

# Only search chemicals not already completed
pending = [
    cid for cid in cids
    if cid not in results
    or results[cid]["STATUS"] == "RETRY"
]

print("Total chemicals:", len(cids))
print("Chemicals remaining:", len(pending))
print("Starting PubChem lookup...")

batch_size = 10

try:
    for start in range(0, len(pending), batch_size):

        batch = pending[start:start + batch_size]

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            "compound/cid/"
            + ",".join(batch)
            + "/synonyms/JSON"
        )

        # Retry temporary connection errors
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                with urllib.request.urlopen(
                    request, timeout=90
                ) as response:
                    data = json.load(response)

                break

            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))

        items = data["InformationList"]["Information"]
        returned = set()

        for item in items:
            cid = str(item["CID"])
            returned.add(cid)

            synonyms = item.get("Synonym", [])

            matches = sorted(set(
                str(name).strip().upper()
                for name in synonyms
                if re.fullmatch(
                    r"DTXSID\d+",
                    str(name).strip(),
                    re.IGNORECASE
                )
            ))

            if len(matches) == 1:
                status = "MATCHED"
                dtxsid = matches[0]
            elif len(matches) > 1:
                status = "MULTIPLE"
                dtxsid = ""
            else:
                status = "NOT_IN_SYNONYMS"
                dtxsid = ""

            results[cid] = {
                "PUBCHEM_CID": cid,
                "DTXSID": dtxsid,
                "CANDIDATES": ";".join(matches),
                "STATUS": status
            }

        # Flag missing responses for retry
        for cid in batch:
            if cid not in returned:
                results[cid] = {
                    "PUBCHEM_CID": cid,
                    "DTXSID": "",
                    "CANDIDATES": "",
                    "STATUS": "RETRY"
                }

        batch_number = start // batch_size + 1

        if batch_number % 5 == 0:
            save_progress()

        if batch_number % 10 == 0:
            print(
                "Processed:",
                min(start + batch_size, len(pending)),
                "of",
                len(pending)
            )

        # Avoid sending requests too quickly
        time.sleep(0.6)

except KeyboardInterrupt:
    print("\nStopped by user. Saving progress.")

except Exception as e:
    print("\nLookup paused due to an error:", e)
    print("Run the script again to resume.")

finally:
    save_progress()

# Add matched identifiers to master
matched = {
    cid: row["DTXSID"]
    for cid, row in results.items()
    if row["STATUS"] == "MATCHED"
}

statuses = {
    cid: row["STATUS"]
    for cid, row in results.items()
}

master["DTXSID"] = master["PUBCHEM_CID"].map(matched)
master["DTXSID_LOOKUP_STATUS"] = (
    master["PUBCHEM_CID"]
    .map(statuses)
    .fillna("NOT_LOOKED_UP")
)

master.to_csv(
    folder / "AR_MASTER_WITH_DTXSID.csv",
    index=False
)

# Preserve all experimental records with IDs
records = pd.read_csv(
    folder / "AR_ALL_EXPERIMENTAL_RECORDS.csv",
    dtype="string"
)

records["DTXSID"] = records["PUBCHEM_CID"].map(matched)

records.to_csv(
    folder / "AR_ALL_RECORDS_WITH_DTXSID.csv",
    index=False
)

print("\n" + "=" * 50)
print("DTXSID LOOKUP SUMMARY")
print("=" * 50)

print("Total chemicals:", len(cids))
print("Matched:", len(matched))

for status, count in (
    master["DTXSID_LOOKUP_STATUS"]
    .value_counts()
    .items()
):
    print(status + ":", count)

print("\nFiles saved in NEW_MASTER folder.")
print("DONE!")
