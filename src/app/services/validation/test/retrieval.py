import json
import time
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src", "app", "services"))

from dotenv import load_dotenv
load_dotenv()

from src.app.core.rag.retrieval import perform_search_with_rerank_hybrid

# Format: (metabolite_name, expected_doi, is_ambiguous)
# is_ambiguous=True means the metabolite appears in multiple papers -> top-10 only

VALIDATION_SET = [
    ("vanillic acid",                        "10.1002/mds.27173", False),
    ("3-hydroxykynurenine",                  "10.1002/mds.27173", False),
    ("isoleucyl-alanine",                    "10.1002/mds.27173", False),
    ("5-acetylamino-6-amino-3-methyluracil", "10.1002/mds.27173", False),
    ("theophylline",                         "10.1002/mds.27173", True),
    ("3-methoxytyrosine",                    "10.1038/s41531-024-00732-z", True),
    ("glycine",                              "10.1038/s41531-024-00732-z", True),
    ("pantothenic acid",                     "10.1038/s41531-024-00732-z", True),
    ("caffeine",                             "10.1038/s41531-024-00732-z", True),
    ("ornithine",                            "10.1038/s41531-024-00732-z", False),
    ("tyrosine",                             "10.1038/s41531-024-00732-z", True),
    ("itaconate",                            "10.1186/s13024-023-00694-5", False),
    ("cysteine-S-sulfate",                   "10.1186/s13024-023-00694-5", False),
    ("p-cresol glucuronide",                 "10.1186/s13024-023-00694-5", True),
    ("1-lyso-2-arachidonoyl-phosphatidate",  "10.1021/acschemneuro.4c00355", False),
    ("ferulic acid",                         "10.1021/acschemneuro.4c00355", False),
    ("hypoxanthine",                         "10.1021/acschemneuro.4c00355", True),
    ("Tropinone",                            "10.1111/cns.70424", False),
    ("N-Acetyl-L-tyrosine",                  "10.1111/cns.70424", False),
    ("Propyl gallate",                       "10.1111/cns.70424", False),
    ("3,4-Dihydroxyphenylglycol O-sulfate",  "10.1111/cns.70424", False),
    ("Isobenzofuranone",                     "10.1111/cns.70424", False),
    ("NP-019988",                            "10.1111/cns.70424", False),
    ("Ethyl N-(1,4-thiazinan-4-ylcarbothioyl) carbamate", "10.1111/cns.70424", False),
    ("Vulgraon B",                           "10.1111/cns.70424", False),
    ("histidine",                            "10.1016/j.brainres.2025.149625", False),
    ("acetate",                              "10.1016/j.brainres.2025.149625", False),
    ("acetoacetate",                         "10.1016/j.brainres.2025.149625", False),
    ("N-acetyl-glycoproteins",               "10.1016/j.brainres.2025.149625", False),
    ("sarcosine",                            "10.1016/j.brainres.2025.149625", False),
    ("L-asparagine",                         "10.1016/j.brainres.2025.149625", False),
    ("trimethylamine",                       "10.1016/j.brainres.2025.149625", False),
    ("3-beta-hydroxybutyrate",               "10.1016/j.brainres.2025.149625", False),
    ("isovaleric acid",                      "10.1016/j.brainres.2025.149625", False),
    ("creatine",                             "10.1016/j.brainres.2025.149625", False),
    ("creatinine",                           "10.1016/j.brainres.2025.149625", False),
    ("choline",                              "10.1016/j.brainres.2025.149625", True),
    ("arginine",                             "10.1016/j.brainres.2025.149625", False),
    ("cysteine",                             "10.1016/j.brainres.2025.149625", False),
    ("urea",                                 "10.1016/j.brainres.2025.149625", False),
    ("glucose",                              "10.1016/j.brainres.2025.149625", True),
    ("glutamine",                            "10.1016/j.brainres.2025.149625", True),
    ("butyrate",                             "10.3389/fnagi.2022.927625", True),
    ("phenylalanine",                        "10.3389/fnagi.2022.927625", True),
    ("butyric acid",                         "10.1038/s41598-026-36756-z", False),
    ("indoleacetic acid",                    "10.1038/s41598-026-36756-z", False),
    ("phosphatidylcholine aa C40:2",         "10.1038/s41598-026-36756-z", False),
    ("acylcarnitine C12:1",                  "10.1038/s41598-026-36756-z", False),
    ("tartronate",                           "10.1002/mds.26992", False),
    ("catechol sulfate",                     "10.1002/mds.26992", False),
    ("hexanoylglutamine",                    "10.1002/mds.26992", False),
    ("decanoylcarnitine",                    "10.1002/mds.26992", False),
    ("myristoleoylcarnitine",                "10.1002/mds.26992", False),
    ("octanoylcarnitine",                    "10.1002/mds.26992", False),
    ("oleoylcarnitine",                      "10.1002/mds.26992", False),
    ("palmitoleoylcarnitine",                "10.1002/mds.26992", False),
    ("suberoylcarnitine",                    "10.1002/mds.26992", False),
    ("octadecanedioate",                     "10.1002/mds.26992", False),
    ("3-hydroxysebacate",                    "10.1002/mds.26992", False),
    ("1-methylhistamine",                    "10.1002/mds.26992", False),
    ("1-myristoyl-GPC",                      "10.1002/mds.26992", False),
    ("2-myristoyl-GPC",                      "10.1002/mds.26992", False),
    ("1,3-dimethylurate",                    "10.1002/mds.26992", False),
    ("oxalate",                              "10.1002/mds.26992", True),
    ("x-12462",                              "10.1002/mds.26992", False),
    ("x-18249",                              "10.1002/mds.26992", False),
    ("x-21735",                              "10.1002/mds.26992", False),
    ("x-23756",                              "10.1002/mds.26992", False),
    ("L-Leucine",                            "10.3390/ijms241612849", True),
    ("N-Acetylserine",                       "10.3390/ijms241612849", False),
    ("L-Tryptophan",                         "10.3390/ijms241612849", True),
    ("N-Acetyl-L-alanine",                   "10.3390/ijms241612849", False),
    ("L-Isoleucine",                         "10.3390/ijms241612849", True),
    ("L-Tyrosine",                           "10.3390/ijms241612849", True),
    ("L-Phenylalanine",                      "10.3390/ijms241612849", True),
    ("3-Hydroxycapric acid",                 "10.3390/ijms241612849", False),
    ("succinic acid",                        "10.3390/ijms241612849", False),
    ("malic acid",                           "10.3390/ijms241612849", False),
    ("citric acid",                          "10.3390/ijms241612849", False),
    ("L-2-Hydroxyglutaric acid",             "10.3390/ijms241612849", False),
    ("3-Hydroxyglutaric acid",               "10.3390/ijms241612849", False),
    ("2,4-Dihydroxybenzoic acid",            "10.3390/ijms241612849", False),
    ("2-Pyrocatechuic acid",                 "10.3390/ijms241612849", False),
    ("adenosine",                            "10.3390/ijms241612849", False),
    ("NAD",                                  "10.3390/ijms241612849", False),
    ("alpha-Ketoisovaleric acid",            "10.3390/ijms241612849", False),
    ("Ketoleucine",                          "10.3390/ijms241612849", False),
    ("Oxoglutaric acid",                     "10.3390/ijms241612849", False),
    ("Trigonelline",                         "10.3390/ijms241612849", True),
    ("hydroxy-CMPF",                         "10.1038/s41598-025-30521-4", False),
    ("Hydroxy-3-carboxy-4-methyl-5-propyl-2-furanpropanoic acid", "10.1038/s41598-025-30521-4", False),
    ("Carnitine C14",                        "10.1038/s41598-025-30521-4", False),
    ("1-dihomo-linolenylglycerol",           "10.1038/s41598-025-30521-4", False),
    ("1-linoleoyl-GPG",                      "10.1038/s41598-025-30521-4", False),
    ("O-sulfo-L-tyrosine",                   "10.1038/s41598-025-30521-4", False),
    ("tryptophan",                           "10.1038/s41598-025-30521-4", True),
    ("N8-acetylspermidine",                  "10.1002/ana.25516", False),
    ("N-acetylputrescine",                   "10.1002/ana.25516", False),
    ("spermine",                             "10.1002/ana.25516", False),
    ("spermidine",                           "10.1002/ana.25516", False),
    ("spermine spermidine ratio",            "10.1002/ana.25516", False),
    ("N1,N8-diacetylspermidine",             "10.1002/ana.25516", False),
    ("paraxanthine",                         "10.1212/WNL.0000000000004888", False),
    ("theobromine",                          "10.1212/WNL.0000000000004888", False),
    ("1,7-dimethyluric acid",                "10.1212/WNL.0000000000004888", False),
    ("1,3,7-trimethyluric acid",             "10.1212/WNL.0000000000004888", False),
    ("1-Methylxanthine",                     "10.1212/WNL.0000000000004888", False),
    ("3-Methylxanthine",                     "10.1212/WNL.0000000000004888", False),
    ("1-Methyluric acid",                    "10.1212/WNL.0000000000004888", False),
    ("7-Methylxanthine",                     "10.1212/WNL.0000000000004888", False),
    ("AFMU",                                 "10.1212/WNL.0000000000004888", False),
    ("AAMU",                                 "10.1212/WNL.0000000000004888", False),
    ("homovanillic acid",                    "10.1002/mds.28608", False),
    ("HVA",                                  "10.1002/mds.28608", False),
    ("3,4-dihydroxyphenylacetic acid",       "10.1002/mds.28608", False),
    ("DOPAC",                                "10.1002/mds.28608", False),
    ("phenylacetyl-L-glutamine",             "10.1186/s13024-021-00425-8", False),
    ("kynurenine",                           "10.1186/s13024-021-00425-8", False),
    ("indolelactic acid",                    "10.1186/s13024-021-00425-8", False),
    ("proline",                              "10.1186/s13024-021-00425-8", False),
    ("p-Cresol glucuronide",                 "10.1186/s13024-021-00425-8", True),
    ("p-Cresol sulfate",                     "10.1186/s13024-021-00425-8", False),
    ("free fatty acids",                     "10.1186/s13024-021-00425-8", False),
    ("acylcarnitines",                       "10.1186/s13024-021-00425-8", True),
    ("Phosphatidylcholines",                 "10.1186/s13024-021-00425-8", True),
    ("Sphingomyelin SM(32:2)",               "10.1186/s13024-021-00425-8", False),
    ("cortisol",                             "10.1186/s13024-021-00425-8", False),
    ("corticosterone",                       "10.1186/s13024-021-00425-8", False),
    ("cis-Aconitic acid",                    "10.1186/s13024-021-00425-8", False),
    ("uridine",                              "10.1186/s13024-021-00425-8", False),
    ("phosphate",                            "10.1186/s13024-021-00425-8", False),
    ("myoinositol",                          "10.1186/s13024-021-00425-8", False),
]

QUERY_TEMPLATES = [
    "What does {metabolite} have to do with Parkinson's disease?",
    "What is known about {metabolite} in Parkinson's disease?",
    "{metabolite} Parkinson's disease",
]

def doi_matches(expected, retrieved_url):
    return expected in retrieved_url

def search_with_best_query(metabolite, expected_doi):
    best_result = None
    best_match_level = -1

    for template in QUERY_TEMPLATES:
        query = template.format(metabolite=metabolite)
        try:
            chunks = perform_search_with_rerank_hybrid(query)
            retrieved_dois = [r['meta']['url'] for r in chunks]

            top1 = retrieved_dois[0] if retrieved_dois else None
            top1_match  = doi_matches(expected_doi, top1) if top1 else False
            top3_match  = any(doi_matches(expected_doi, d) for d in retrieved_dois[:3])
            top10_match = any(doi_matches(expected_doi, d) for d in retrieved_dois[:10])

            if top1_match:
                match_level = 2
            elif top3_match:
                match_level = 1
            elif top10_match:
                match_level = 0
            else:
                match_level = -1

            if match_level > best_match_level:
                best_match_level = match_level
                best_result = {
                    "query_used": query,
                    "top1_doi": top1,
                    "top1_match": top1_match,
                    "top3_match": top3_match,
                    "top10_match": top10_match,
                    "retrieved_dois": retrieved_dois[:10],
                }

            if top1_match:
                break

        except Exception as e:
            print(f"    ⚠️  Query failed: {query[:50]}... → {e}")
            continue

    return best_result

def run_validation():
    results = []
    total = len(VALIDATION_SET)

    print(f"Running {total} validation queries (up to {len(QUERY_TEMPLATES)} phrasings each)...\n")

    for i, (metabolite, expected_doi, is_ambiguous) in enumerate(VALIDATION_SET):
        label = "[AMBIGUOUS]" if is_ambiguous else ""
        print(f"[{i+1}/{total}] {label} {metabolite}...")

        best = search_with_best_query(metabolite, expected_doi)

        if best is None:
            print(f"  ❌ ERROR: all queries failed")
            results.append({
                "metabolite":   metabolite,
                "expected_doi": expected_doi,
                "is_ambiguous": is_ambiguous,
                "error":        "all queries failed",
                "top1_match":   False,
                "top3_match":   False,
                "top10_match":  False,
            })
            continue

        top1_match  = best["top1_match"]
        top3_match  = best["top3_match"]
        top10_match = best["top10_match"]

        if is_ambiguous:
            icon   = "✅" if top10_match else "❌"
            detail = f"top10={'✅' if top10_match else '❌'} (ambiguous)"
        else:
            if top1_match:    icon, detail = "✅", "top1=✅"
            elif top3_match:  icon, detail = "⚠️ ", "top1=❌ top3=✅"
            elif top10_match: icon, detail = "⚠️ ", "top1=❌ top3=❌ top10=✅"
            else:             icon, detail = "❌", "top1=❌ top3=❌ top10=❌"

        print(f"  {icon} {detail} | expected={expected_doi} | got={best['top1_doi']}")
        print(f"       query: \"{best['query_used'][:70]}\"")

        results.append({
            "metabolite":     metabolite,
            "expected_doi":   expected_doi,
            "is_ambiguous":   is_ambiguous,
            "query_used":     best["query_used"],
            "top1_doi":       best["top1_doi"],
            "top1_match":     top1_match,
            "top3_match":     top3_match,
            "top10_match":    top10_match,
            "retrieved_dois": best["retrieved_dois"],
        })

        time.sleep(0.1)

    unambiguous = [r for r in results if not r.get("is_ambiguous")]
    ambiguous   = [r for r in results if r.get("is_ambiguous")]

    u_top1  = sum(1 for r in unambiguous if r.get("top1_match"))
    u_top3  = sum(1 for r in unambiguous if r.get("top3_match"))
    u_top10 = sum(1 for r in unambiguous if r.get("top10_match"))
    a_top10 = sum(1 for r in ambiguous   if r.get("top10_match"))

    print(f"\n{'='*60}")
    print(f"RETRIEVAL VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Unambiguous metabolites ({len(unambiguous)} queries):")
    print(f"  Top-1  accuracy: {u_top1}/{len(unambiguous)}  ({100*u_top1/max(len(unambiguous),1):.1f}%)")
    print(f"  Top-3  accuracy: {u_top3}/{len(unambiguous)}  ({100*u_top3/max(len(unambiguous),1):.1f}%)")
    print(f"  Top-10 accuracy: {u_top10}/{len(unambiguous)} ({100*u_top10/max(len(unambiguous),1):.1f}%)")
    print(f"\nAmbiguous metabolites ({len(ambiguous)} queries, top-10 only):")
    print(f"  Top-10 accuracy: {a_top10}/{len(ambiguous)}  ({100*a_top10/max(len(ambiguous),1):.1f}%)")

    failures = [r for r in results if "error" not in r and not r["top10_match"]]
    if failures:
        print(f"\nFailed queries (not found in top-10 across all phrasings):")
        for r in failures:
            print(f"  ❌ {r['metabolite']}")
            print(f"     expected: {r['expected_doi']}")
            print(f"     got:      {r.get('top1_doi', '—')}")

    output = {
        "summary": {
            "query_templates": QUERY_TEMPLATES,
            "unambiguous": {
                "total":          len(unambiguous),
                "top1_accuracy":  round(u_top1  / max(len(unambiguous), 1), 3),
                "top3_accuracy":  round(u_top3  / max(len(unambiguous), 1), 3),
                "top10_accuracy": round(u_top10 / max(len(unambiguous), 1), 3),
            },
            "ambiguous": {
                "total":          len(ambiguous),
                "top10_accuracy": round(a_top10 / max(len(ambiguous), 1), 3),
            }
        },
        "results": results
    }

    with open("validation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to validation_results.json")

if __name__ == "__main__":
    run_validation()