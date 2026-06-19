from app.core.etl.obtain_ids import fetch_paper_metadata, get_pmcid_from_pmid
import json, time

VALIDATION_PMIDS = [
    "38115046",
    "39177430",
    "40376890",
    "35959296",
    "41565969",
    "37629028",
    "41872227",
    "31155745",
    "29298852",
    "33942926"
]

metadata = []
for pmid in VALIDATION_PMIDS:
    entry = fetch_paper_metadata(pmid)
    if entry:
        entry["pmcid"] = get_pmcid_from_pmid(pmid)
        metadata.append(entry)
        time.sleep(0.4)

with open("data/metadata_validation.json", "w") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)