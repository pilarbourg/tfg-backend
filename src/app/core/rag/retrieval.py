from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from app.services.db.connection import get_db_connection

embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
ranker = Ranker()

def _db(query, params):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

def vector_search(query, top_k=50):
    query_vector = embedding_model.encode(query).tolist()
    sql = f"""
        SELECT title, source_url, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM limited_papers
        ORDER BY similarity DESC
        LIMIT %s;
    """
    rows = _db(sql, (str(query_vector), top_k))
    return [
        {"text": r[2], "meta": {"title": r[0], "url": r[1]}, "score": float(r[3])}
        for r in rows
    ]

def keyword_search(query, top_k=20):
    sql = f"""
        SELECT content, source_url, title,
               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
        FROM limited_papers
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s;
    """
    rows = _db(sql, (query, query, top_k))
    return [
        {"text": r[0], "meta": {"url": r[1], "title": r[2]}, "score": float(r[3])}
        for r in rows
    ]

def hybrid_search(query):
    candidates = (
        vector_search(query, top_k=50)
        + keyword_search(query, top_k=20)
    )

    seen, unified = set(), []
    for idx, passage in enumerate(candidates):
        snippet = passage["text"][:100]
        if snippet not in seen:
            seen.add(snippet)
            passage["id"] = idx
            unified.append(passage)

    if not unified:
        return []

    reranked = ranker.rerank(RerankRequest(query=query, passages=unified))
    return reranked[:10]

def get_context(user_query):
    context_chunks = hybrid_search(user_query)
    context_text = "".join(
        f"\n--- {res['meta']['url'].replace('_abs', '')} ---\n"
        f"Title: {res['meta']['title']}\n"
        f"DOI: {res['meta']['url'].replace('_abs', '')}\n"
        f"Content: {res['text']}\n"
        for res in context_chunks
    )
    return context_chunks, context_text