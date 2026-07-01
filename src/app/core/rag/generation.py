import os
import time
from dotenv import load_dotenv
from .retrieval import hybrid_search
from langchain_ollama import ChatOllama

load_dotenv()

local_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.0,
    num_ctx=8192,
)

def get_atlas_stream(user_query, context_text):
    system_prompt = f"""
    You are a research assistant. Answer using ONLY the research excerpts below.
    Each excerpt is labeled with its DOI.

    Rules:
    - If the excerpts contain the answer, state it directly and cite the DOI of 
    the excerpt you used, in parentheses.
    - If the excerpts contain nothing relevant to the question, reply with exactly:
    "This information is not present in the retrieved literature."
    - Never do both. Either answer, or give that exact sentence — not both.
    - Only use DOIs that appear in the excerpts. Never invent one.
    - Write 1-3 sentences of plain scientific prose. No lists.

    Example 1 — Single metabolite, mechanistic question:
    User: What motor symptoms does PD manifest with?
    Atlas: PD manifests with typical motor symptoms: rest tremor, muscle rigidity, bradykinesia, postural instability, limb rigidity, and akinesia (10.1016/j.neubiorev.2025.106310).

    CONTEXT:
    {context_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    total_chars = len(system_prompt) + len(user_query)
    print(f"Total prompt length: {total_chars} chars")
    print(context_text)

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
    context_chunks = hybrid_search(user_query)
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
