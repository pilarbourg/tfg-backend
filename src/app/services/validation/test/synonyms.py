import xml.etree.ElementTree as ET

NS = "http://www.hmdb.ca"

def build_synonym_map(xml_path, target_ids):
    target_ids = set(target_ids)
    results = {}
    context = ET.iterparse(xml_path, events=("end",))

    for event, elem in context:
        if elem.tag == f"{{{NS}}}metabolite":
            accession = elem.findtext(f"{{{NS}}}accession")
            if accession in target_ids:
                synonym_elements = elem.find(f"{{{NS}}}synonyms")
                if synonym_elements is not None:
                    results[accession] = [
                        s.text for s in synonym_elements.findall(f"{{{NS}}}synonym")
                        if s.text
                    ]
                else:
                    results[accession] = []
                    
                if len(results) == len(target_ids):
                    break
            elem.clear()

    missing = target_ids - set(results.keys())
    if missing:
        print(f"Warning: {len(missing)} IDs not found: {missing}")

    return results


hmdb_ids = [
  "HMDB0000484",
  "HMDB0000732",
  "HMDB0028900",
  "HMDB0004400",
  "HMDB0001889",
]

synonym_map = build_synonym_map("/Users/pilarbourg/Downloads/hmdb_metabolites.xml", hmdb_ids)

for hmdb_id, synonyms in synonym_map.items():
    print(f"\n{hmdb_id} ({len(synonyms)} synonyms):")
    print("; ".join(synonyms))