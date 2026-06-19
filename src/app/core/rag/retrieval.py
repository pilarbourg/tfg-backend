from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from app.services.db.connection import get_db_connection

embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
ranker = Ranker()

ALLOWED_TABLES = {"research_papers", "validation_papers"}


def _validate_table(target_table):
    if target_table not in ALLOWED_TABLES:
        raise ValueError(f"Unauthorized database target: {target_table}")


def _execute_query(query, params):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def perform_vector_search(query, target_table="validation_papers", top_k=50):
    _validate_table(target_table)
    query_vector = embedding_model.encode(query).tolist()
    sql = f"""
        SELECT title, source_url, content, 1 - (embedding <=> %s::vector) AS similarity
        FROM {target_table}
        ORDER BY similarity DESC
        LIMIT %s;
    """
    rows = _execute_query(sql, (str(query_vector), top_k))
    return [
        {"text": r[2], "meta": {"title": r[0], "url": r[1]}, "score": float(r[3])}
        for r in rows
    ]


def keyword_search_enterprise(query, target_table="validation_papers", top_k=20):
    _validate_table(target_table)
    sql = f"""
        SELECT content, source_url, title,
               ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank
        FROM {target_table}
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s;
    """
    rows = _execute_query(sql, (query, query, top_k))
    return [
        {"text": r[0], "meta": {"url": r[1], "title": r[2]}, "score": float(r[3])}
        for r in rows
    ]


def perform_search_with_rerank_hybrid(query, target_table="validation_papers"):
    candidates = (
        perform_vector_search(query, target_table=target_table, top_k=50)
        + keyword_search_enterprise(query, target_table=target_table, top_k=20)
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


def get_context(user_query, target_table="validation_papers"):
    context_chunks = perform_search_with_rerank_hybrid(user_query, target_table=target_table)
    context_text = "".join(
        f"\n--- {res['meta']['url'].replace('_abs', '')} ---\n"
        f"Title: {res['meta']['title']}\n"
        f"DOI: {res['meta']['url'].replace('_abs', '')}\n"
        f"Content: {res['text']}\n"
        for res in context_chunks
    )
    return context_chunks, context_text