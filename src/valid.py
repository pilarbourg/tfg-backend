import json
import matplotlib.pyplot as plt
import numpy as np

with open("synonym_validation_results.json") as f:
    results = json.load(f)

naming_types = ["common", "iupac", "alternative"]
labels = ["Common Name", "IUPAC Name", "Alternative Name"]

kw_hits = []
vec_hits = []
for nt in naming_types:
    group = [r for r in results if r["naming_type"] == nt]
    kw_hits.append(sum(1 for r in group if r["keyword_rank"]))
    vec_hits.append(sum(1 for r in group if r["vector_rank"]))

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars_kw = ax.bar(x - width/2, kw_hits, width, label="Keyword search", color="#298c8c")
bars_vec = ax.bar(x + width/2, vec_hits, width, label="Vector search", color="#9fc8c8")

ax.bar_label(bars_kw, padding=5, fontsize=16, fontweight="bold")
ax.bar_label(bars_vec, padding=5, fontsize=16, fontweight="bold")

ax.set_ylabel("Successful retrievals (out of 5)", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
ax.tick_params(axis="y", labelsize=13)
ax.set_ylim(0, 6)
ax.set_yticks(range(0, 6))
ax.legend(loc="upper right", fontsize=14)
ax.set_title("Source paper retrieval success by metabolite name", fontsize=20, pad=15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("synonym_validation_summary.png", dpi=300, bbox_inches="tight")
plt.show()