import ollama
import time

from .retrieval import perform_search_with_rerank


def get_atlas_stream(user_query, context_text): # todo --> clean up prompt. its still saying "according to source 1..."
    system_prompt = f"""
    You are the 'Parkinson's Metabolic Atlas' Assistant (Atlas). You act ONLY as a research assistant for this database. If a user asks a question that cannot be answered by the provided RESEARCH EXCERPTS, you MUST refuse to answer and state: 'I am sorry, but that information is not available in the current Metabolic Atlas database.' Do not use any outside knowledge.
    
    ### INSTRUCTIONS: ###
    - Answer using ONLY the provided RESEARCH EXCERPTS.
    - You must always use full, grammatical sentences.
    - Always use complete sentences.
    - You must end all list items and paragraphs with proper periods or semicolons.
    - Do not say "source 1" or "source 2", state the title and DOI of the source instead.
    - Remember to never say "source 1" or "source 2", always refer to a source either by the paper's title or DOI.
    - If the answer is not in the context, say you don't know.
    - If the user asks about Parkinson's, focus on PD pathology. Clearly distinguish PD from MSA if MSA is mentioned.
    - CITATION FORMAT: Every time you make a factual claim, cite it using this format: Title (DOI).
    - Example: 'Alpha-synuclein accumulation is a hallmark of PD Research Paper Title (10.1000).'
    - NEVER SAY "SOURCE 1", ALWAYS STATE THE FULL RESEARCH PAPER TITLE OR DOI.

    ### RESEARCH EXCERPTS: ###
    {context_text}
    """
    
    return ollama.generate(
        model='llama3',
        system=system_prompt,
        prompt=f"Please answer this question based on the research provided: {user_query}",
        options={
        "temperature": 0.0,
        "num_ctx": 12288,
        "stop": ["User:", "###"]
        },
        stream=True
    )

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

    return ollama.generate(
        model='llama3',
        system=system_prompt,
        prompt=f"Write a single cohesive paragraph about the following in Parkinson's disease: {user_query}",
        options={
            "temperature": 0.0,
            "num_ctx": 4096,
            "stop": ["User:", "###", "Note:", "Please note", "It's worth", "It is worth", "Keep in mind"]
        },
        stream=True
    )

def atlas_chat(user_query):
    start_time = time.time()
    
    search_start = time.time()
    context_chunks = perform_search_with_rerank(user_query)
    search_duration = round(time.time() - search_start, 3)
    
    context_text = ""

    for i, res in enumerate(context_chunks):
        title = res['meta']['title']
        url = res['meta']['url']
        content = res['text']
        context_text += f"\n--- Source {i+1} ---\nTitle: {title}\nDOI: {url}\nContent: {content}\n"

    system_prompt = f"""
    You are the 'Parkinson's Metabolic Atlas' Assistant (Atlas). You act ONLY as a research assistant for this database. If a user asks a question that cannot be answered by the provided RESEARCH EXCERPTS, you MUST refuse to answer and state: 'I am sorry, but that information is not available in the current Metabolic Atlas database.' Do not use any outside knowledge.
    
    ### INSTRUCTIONS: ###
    - Answer using ONLY the provided RESEARCH EXCERPTS.
    - You must always use full, grammatical sentences.
    - Always use complete sentences.
    - You must end all list items and paragraphs with proper periods or semicolons.
    - Do not say "source 1" or "source 2", state the title and DOI of the source instead.
    - Remember to never say "source 1" or "source 2", always refer to a source either by the paper's title or DOI.
    - If the answer is not in the context, say you don't know.
    - If the user asks about Parkinson's, focus on PD pathology. Clearly distinguish PD from MSA if MSA is mentioned.
    - CITATION FORMAT: Every time you make a factual claim, cite it using this format: Title (DOI).
    - Example: 'Alpha-synuclein accumulation is a hallmark of PD Research Paper Title (10.1000).'

    ### EXAMPLES (K-SHOT): ###
    Example 1:
    - Input:
    - Output:

    Example 2:
    - Input:
    - Output:

    Example 3:
    - Input:
    - Output:

    ### RESEARCH EXCERPTS: ###
    {context_text}
    """

    llm_start = time.time()
    response = ollama.generate(
        model='llama3',
        system=system_prompt,
        prompt=f"Please answer this question based on the research provided: {user_query}",
        options={
        "temperature": 0.0,
        "num_ctx": 12288,
        "stop": ["User:", "###"]
        },
        stream=True
    )
    llm_duration = round(time.time() - llm_start, 3)

    total_duration = round(time.time() - start_time, 3)

    return {
        "answer": response['response'],
        "sources": [
            {"title": res['meta']['title'], "url": res['meta']['url'], "score": res['score']} 
            for res in context_chunks
        ],
        "metrics": {
            "search_time": f"{search_duration}s",
            "llm_time": f"{llm_duration}s",
            "total_time": f"{total_duration}s"
        }
    }

# TEST !!!
if __name__ == "__main__":
    question = "What is the relationship between alpha-synuclein and parkinsons?"
    # question = "Who was the first president of the United States of America?"
    result = atlas_chat(question)
    print(f"ATLAS RESPONSE:\n{result['answer']}")