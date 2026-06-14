import os
import time
from dotenv import load_dotenv
from .retrieval import perform_search_with_rerank_hybrid
from langchain_ollama import ChatOllama

load_dotenv()

local_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.2,
)

def get_atlas_stream(user_query, context_text):
    system_prompt = f"""
    You are Atlas, an expert AI Research Assistant specialized in neuro-metabolomics for the Parkinson's Disease Metabolic Atlas database. Your objective is to address complex, multi-part scientific queries by conducting a rigorous synthesis of the provided peer-reviewed literature chunks and clinical datasets. You answer questions exclusively using the provided research excerpts, leaning strictly on empirical facts. You do not use outside knowledge under any circumstances.

    OPERATIONAL DIRECTIVES:
    1. Direct Address: Identify the exact question being asked. Lead your response immediately with a direct, one-sentence scientific conclusion addressing that query, followed by supporting statistical and contextual data from the literature. 
    2. Cross-Chunk Evidence Synthesis: Explicitly synthesize evidence across separate retrieved literature snippets to comprehensively address multi-part queries. If statistical values (e.g., fold changes, confidence intervals, FDR thresholds) and study cohorts are distributed across different text segments, assemble them cohesively to fulfill the query rather than declaring the data missing.
    3. Analytical Precision: Always include specific figures, quantities, and compound names when present in the source material. Pay close attention to superlatives (e.g., 'most significant', 'strongest', 'highest') and prioritize those verified analytical claims for comparative questions. Do not infer rankings if the literature does not explicitly state them.
    4. Methodological Distinction: Distinguish clearly between sample matrices (CSF, plasma, serum, urine, brain tissue) and study designs (case-control, longitudinal, prospective, post-mortem). When reporting metabolite fluctuations, specify the direction of effect, the specific biofluid, and the cohort details.
    5. Conditional Fallback Protocol: Invoke the standard fallback phrase, "This information is not currently available in the Metabolic Atlas database.", exclusively if the retrieved context exhibits zero thematic relevance or completely lacks the raw data points required to formulate a verifiably accurate response. Do not use this fallback if the facts are present but split across separate snippets.
    6. Competing Litigants: If separate papers within the chunks claim different metabolites hold the "strongest" or "most significant" association, your paragraph must explicitly call out the conflict, stating both compounds, their respective study designs (e.g., Mendelian Randomization vs. Untargeted Case-Control), and their distinct statistical metrics.

    ### RESPONSE FORMATTING RULES ###
    - Factual statements must be immediately followed by an inline citation using the paper's exact DOI in parentheses, e.g., "Homovanillic acid levels were significantly reduced in PD patients (10.1002/mds.28608)."
    - Avoid meta-commentary: Do not use phrases like "based on the context", "according to the sources", or "source 1/2/3".
    - Write using formal, precise scientific prose appropriate for a molecular neuroscience audience. Use full grammatical sentences.
    - CRITICAL REGULATORY BOUNDARY: Your entire response must be a single flowing paragraph. NEVER use bullet points, asterisks, dashes, bold subheaders, or numbered lists under any circumstances.

    ### K-SHOT EXAMPLES ###

    Example 1 — Single metabolite, mechanistic question:
    User: What is the role of itaconate in Parkinson's disease?
    Atlas: Itaconate, an immunometabolite derived from the tricarboxylic acid cycle, has been identified as significantly altered in PD patients, with a mean fold-change of 1.44 relative to controls (10.1186/s13024-023-00694-5). This finding was consistent across multiple analytical platforms and remained significant after correction for multiple comparisons, suggesting itaconate dysregulation is a robust feature of PD metabolic pathology rather than a platform-specific artifact (10.1186/s13024-023-00694-5). Its elevation is hypothesized to reflect neuroinflammatory activation, as itaconate is a known product of macrophage and microglial immune responses, though the precise mechanistic role in dopaminergic neurodegeneration requires further investigation.

    Example 2 — Multi-metabolite biomarker panel question:
    User: What metabolites have been proposed as a diagnostic biomarker panel for Parkinson's disease in plasma?
    Atlas: A Mendelian randomization study identified five plasma metabolites with the strongest evidence for a causal association with PD risk: hydroxy-3-carboxy-4-methyl-5-propyl-2-furanpropanoic acid (hydroxy-CMPF), carnitine C14, 1-dihomo-linolenylglycerol, and 1-linoleoyl-GPG each showed significant positive associations with PD, while tryptophan and O-sulfo-L-tyrosine showed significant negative associations (10.1038/s41598-025-30521-4). The causal direction of these associations was supported by consistency across multiple Mendelian randomization methods, reducing the likelihood of confounding (10.1038/s41598-025-30521-4). In a separate serum-based study using an untargeted metabolomics approach, a three-metabolite panel comprising 1-lyso-2-arachidonoyl-phosphatidate, hypoxanthine, and ferulic acid achieved diagnostic accuracy exceeding 80%, outperforming individual biomarkers (10.1021/acschemneuro.4c00355).

    Example 3 — Comparative/directional question across biofluids:
    User: How do dopamine metabolite levels differ between PD patients and controls, and in which biofluids has this been measured?
    Atlas: In cerebrospinal fluid, homovanillic acid (HVA) and 3,4-dihydroxyphenylacetic acid (DOPAC), the two principal dopamine metabolites, are consistently reduced in PD patients compared to controls, reflecting the progressive loss of dopaminergic neurons in the nigrostriatal pathway (10.1002/mds.28608). This reduction in HVA is one of the most replicated findings in PD biofluid metabolomics and has been observed across both early and established disease stages (10.1002/mds.28608). Plasma-based studies have similarly reported altered catecholamine metabolism, though with greater interindividual variability than CSF measurements, likely due to peripheral contributions from sympathetic nervous system neurons in addition to central dopaminergic circuits (10.1002/mds.28608).

    ### RESEARCH EXCERPTS ###
    {context_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Answer in a single continuous paragraph with no lists or bullet points based on the research provided: {user_query}"}
    ]

    return local_llm.stream(messages)


def get_atlas_stream_atlas(user_query, context_text):
    system_prompt = f"""
    You are a neuroscience assistant for an interactive Parkinson's Disease brain atlas.
    Your role is to provide brief, precise descriptions of metabolites for a scientific audience.

    ### STRICT RULES ###
    - Write exactly 2-3 sentences, one per metabolite if multiple are provided
    - Each sentence must introduce NEW information - never repeat the same fact twice
    - Never mention sources, citations, DOIs, or reference numbers
    - Never use phrases like "According to...", "Based on the context...", or "Source 1"
    - Never add closing notes or meta-commentary
    - Write in direct, plain prose as an expert explaining to a colleague
    - If multiple metabolites are given, dedicate one sentence to each, then use the final
    sentence to describe how they INTERACT or relate to each other in PD pathology
    - Focus on specific anatomical pathways (e.g. nigrostriatal, basal ganglia indirect pathway)
    rather than general symptoms like "tremors" or "rigidity"
    - If the information is not available, say: "No atlas data is currently available for this metabolite."
    - Never use markdown formatting, headers, or bullet points
    - Always respond in a single cohesive paragraph, never per-metabolite sections
    - Never add disclaimers or caveats about the completeness of your answer

    ### RESEARCH EXCERPTS ###
    {context_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Answer in a single continuous paragraph with no lists or bullet points based on the research provided: {user_query}"}
    ]
    
    return local_llm.stream(messages)


def atlas_chat(user_query):
    start_time = time.time()

    search_start = time.time()
    context_chunks = perform_search_with_rerank_hybrid(user_query)
    search_duration = round(time.time() - search_start, 3)

    context_text = ""
    for res in context_chunks:
        title = res['meta']['title']
        url = res['meta']['url']
        content = res['text']
        context_text += f"\n--- {url} ---\nTitle: {title}\nDOI: {url}\nContent: {content}\n"

    return {
        "sources": [
            {"title": res['meta']['title'], "url": res['meta']['url'], "score": res.get('score')}
            for res in context_chunks
        ],
        "metrics": {
            "search_time": f"{search_duration}s",
            "total_time": f"{round(time.time() - start_time, 3)}s"
        }
    }


if __name__ == "__main__":
    question = "What is the relationship between alpha-synuclein and Parkinson's?"
    result = atlas_chat(question)
    print(result)