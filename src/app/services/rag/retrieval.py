from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from db.connection import get_db_connection

embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
ranker = Ranker()

def get_context(user_query):
    context_chunks = perform_search_with_rerank_hybrid(user_query)
    context_text = ""

    for i, res in enumerate(context_chunks):
        title = res['meta']['title']
        url = res['meta']['url']
        content = res['text']
        context_text += f"\n--- {url} ---\nTitle: {title}\nDOI: {url}\nContent: {content}\n"

    return context_chunks, context_text

def perform_search(query, top_k=10):
    conn = get_db_connection()
    cur = conn.cursor()
    
    query_vector = embedding_model.encode(query).tolist()
    
    search_query = """
        SELECT title, source_url, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM research_papers
        ORDER BY similarity DESC
        LIMIT %s;
    """
    
    cur.execute(search_query, (str(query_vector), top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def perform_search_with_rerank(query, top_k=50):
    initial_results = perform_search(query, top_k=50)
    passages = [
        {"id": i, "text": res[2], "meta": {"title": res[0], "url": res[1]}} 
        for i, res in enumerate(initial_results)
    ]
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    return results[:10]

def keyword_search(query, conn, top_k=5):
    import re
    match = re.search(r"about (.+?) in Parkinson", query, re.IGNORECASE)
    if not match:
        return []
    term = match.group(1).strip()
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content, source_url, title
            FROM research_papers
            WHERE content ILIKE %s
            LIMIT %s
        """, (f"%{term}%", top_k))
        rows = cur.fetchall()
    
    return [{"text": r[0], "meta": {"url": r[1], "title": r[2]}, "score": 0.5} for r in rows]

def perform_search_with_rerank_hybrid(query):
    vector_results = perform_search_with_rerank(query)
    conn = get_db_connection()
    try:
        keyword_results = keyword_search(query, conn)
    finally:
        conn.close()

    seen = set()
    merged = []
    for r in vector_results + keyword_results:
        url = r["meta"]["url"]
        if url not in seen:
            seen.add(url)
            merged.append(r)
    return merged