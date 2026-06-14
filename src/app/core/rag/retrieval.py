from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from app.services.db.connection import get_db_connection

embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
ranker = Ranker()

def get_context(user_query, target_table="research_papers"):
    """
    Unified entry point for retrieving context. 
    Set target_table="validation_papers" when running validation scripts.
    """
    context_chunks = perform_search_with_rerank_hybrid(user_query, target_table=target_table)
    context_text = ""

    for i, res in enumerate(context_chunks):
        title = res['meta']['title']
        url = res['meta']['url'].replace('_abs', '')
        content = res['text']
        context_text += f"\n--- {url} ---\nTitle: {title}\nDOI: {url}\nContent: {content}\n"

    return context_chunks, context_text

def perform_vector_search(query, target_table="research_papers", top_k=50):
    """Fetches a broad pool of vector candidates from the specified target table."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    query_vector = embedding_model.encode(query).tolist()
    
    # Secure table identifier interpolation matching enterprise database practices
    allowed_tables = {"research_papers", "validation_papers"}
    if target_table not in allowed_tables:
        raise ValueError(f"Unauthorized database target: {target_table}")
        
    search_query = f"""
        SELECT title, source_url, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM {target_table}
        ORDER BY similarity DESC
        LIMIT %s;
    """
    
    cur.execute(search_query, (str(query_vector), top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return [
        {"text": res[2], "meta": {"title": res[0], "url": res[1]}, "score": float(res[3])}
        for res in results
    ]

def keyword_search_enterprise(query, target_table="research_papers", top_k=20):
    """
    Enterprise native Full-Text Search using PostgreSQL GIN indexing.
    Eliminates manual application-side text preprocessing.
    """
    allowed_tables = {"research_papers", "validation_papers"}
    if target_table not in allowed_tables:
        raise ValueError(f"Unauthorized database target: {target_table}")

    conn = get_db_connection()
    cur = conn.cursor()
    
    # Utilizes native plainto_tsquery for structural linguistic stemming
    search_query = f"""
        SELECT content, source_url, title,
               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank
        FROM {target_table}
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s;
    """
    
    cur.execute(search_query, (query, query, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    return [
        {"text": r[0], "meta": {"url": r[1], "title": r[2]}, "score": float(r[3])} 
        for r in rows
    ]

def perform_search_with_rerank_hybrid(query, target_table="research_papers"):
    """Executes parallel global retrieval passes and ranks inputs via FlashRank."""
    vector_candidates = perform_vector_search(query, target_table=target_table, top_k=50)
    keyword_candidates = keyword_search_enterprise(query, target_table=target_table, top_k=20)
    
    seen_content = set()
    unified_passages = []
    
    # Dynamic de-duplication loop
    for idx, passage in enumerate(vector_candidates + keyword_candidates):
        content_snippet = passage["text"][:100] 
        if content_snippet not in seen_content:
            seen_content.add(content_snippet)
            passage["id"] = idx 
            unified_passages.append(passage)
            
    if not unified_passages:
        return []

    rerank_request = RerankRequest(query=query, passages=unified_passages)
    reranked_results = ranker.rerank(rerank_request)
    
    return reranked_results[:10]