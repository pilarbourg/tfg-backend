from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from db.connection import get_db_connection

embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
ranker = Ranker()

def get_context(user_query):
    context_chunks = perform_search_with_rerank(user_query)
    
    context_text = ""
    for i, res in enumerate(context_chunks):
        context_text += f"\n--- Source {i+1} ---\nTitle: {res['meta']['title']}\nDOI: {res['meta']['url']}\nContent: {res['text']}\n"
    
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

def perform_search_with_rerank(query, top_k=20):
    initial_results = perform_search(query, top_k=20) 
    
    passages = [
        {"id": i, "text": res[2], "meta": {"title": res[0], "url": res[1]}} 
        for i, res in enumerate(initial_results)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    return results[:10] # returns 5-10 highest quality text from top 20 !!!
