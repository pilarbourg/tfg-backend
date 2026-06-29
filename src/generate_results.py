import json
from pathlib import Path
from app.core.rag.retrieval import hybrid_search
from app.core.rag.generation import get_atlas_stream

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "validation_dataset.json"

with open(DATASET_PATH) as f:
    questions = json.load(f)["questions"]

results = []
for q in questions:
    query = q["sample_query"]
    chunks = hybrid_search(query)
    context_text = "\n".join(f"DOI: {c['meta']['url']}\nContent: {c['text']}" for c in chunks)
    answer = "".join(c.content for c in get_atlas_stream(query, context_text))
    results.append({
        "user_input": query,
        "retrieved_contexts": [c["text"] for c in chunks],
        "response": answer,
        "reference": q["expected_output"],
    })

with open("rag_results.json", "w") as f:
    json.dump(results, f, indent=2)