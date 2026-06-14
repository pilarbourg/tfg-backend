from fastapi import APIRouter, Depends
from psycopg2.extensions import connection
from app.api.dependencies import get_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
def get_dashboard_stats(conn: connection = Depends(get_db)):
    """
    Returns corpus statistics including paper counts, chunk distribution,
    publisher breakdown, and recently ingested documents.
    """
    with conn.cursor() as cur:

        cur.execute("""
            SELECT COUNT(DISTINCT REPLACE(source_url, '_abs', ''))
            FROM research_papers
        """)
        total_papers = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT source_url)
            FROM research_papers
            WHERE source_url NOT LIKE '%_abs'
        """)
        full_text_count = cur.fetchone()[0]

        abstract_only = total_papers - full_text_count

        cur.execute("SELECT COUNT(*) FROM research_papers")
        total_chunks = cur.fetchone()[0]

        cur.execute("""
            SELECT ROUND(AVG(chunk_count))
            FROM (
                SELECT source_url, COUNT(*) as chunk_count
                FROM research_papers
                WHERE source_url NOT LIKE '%_abs'
                GROUP BY source_url
            ) sub
        """)
        avg_chunks = cur.fetchone()[0] or 0

        cur.execute("""
            SELECT
                SPLIT_PART(
                    REPLACE(source_url, '_abs', ''),
                    '/', 1
                ) as publisher,
                COUNT(DISTINCT REPLACE(source_url, '_abs', '')) as count
            FROM research_papers
            GROUP BY publisher
            ORDER BY count DESC
            LIMIT 8
        """)
        publishers = [
            {"publisher": row[0], "count": row[1]}
            for row in cur.fetchall()
        ]

        cur.execute("""
            SELECT
                CASE
                    WHEN chunk_count BETWEEN 1 AND 5 THEN '1-5'
                    WHEN chunk_count BETWEEN 6 AND 20 THEN '6-20'
                    WHEN chunk_count BETWEEN 21 AND 50 THEN '21-50'
                    ELSE '50+'
                END as range,
                COUNT(*) as papers
            FROM (
                SELECT source_url, COUNT(*) as chunk_count
                FROM research_papers
                WHERE source_url NOT LIKE '%_abs'
                GROUP BY source_url
            ) sub
            GROUP BY range
            ORDER BY MIN(chunk_count)
        """)
        chunk_dist = [
            {"range": row[0], "papers": row[1]}
            for row in cur.fetchall()
        ]

        cur.execute("""
            SELECT
                title,
                REPLACE(source_url, '_abs', '') as doi,
                MAX(ingested_at) as ingested_at,
                COUNT(*) as chunk_count
            FROM research_papers
            GROUP BY title, doi
            ORDER BY MAX(ingested_at) DESC
            LIMIT 50
        """)
        recent_docs = [
            {
                "title": row[0],
                "doi": row[1],
                "ingested_at": str(row[2]) if row[2] else None,
                "chunks": row[3],
            }
            for row in cur.fetchall()
        ]

    return {
        "total_papers": total_papers,
        "full_text_count": full_text_count,
        "abstract_only": abstract_only,
        "total_chunks": total_chunks,
        "avg_chunks_per_paper": int(avg_chunks),
        "publishers": publishers,
        "chunk_distribution": chunk_dist,
        "recent_docs": recent_docs,
    }


@router.get("/keywords")
def get_keywords(conn: connection = Depends(get_db)):
    """
    Returns the most frequent domain-specific keywords across the corpus.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT word, COUNT(*) as freq
            FROM (
                SELECT regexp_split_to_table(
                    lower(content), '[^a-z]+'
                ) as word
                FROM research_papers
                LIMIT 10000
            ) words
            WHERE length(word) > 5
            AND word NOT IN (
                'parkinson','disease','patients','studies',
                'results','methods','study','analysis',
                'using','based','within','between','these',
                'their','which','shown','found','after',
                'clinical','however','including','associated',
                'significant','compared','research','also',
                'other','further','brain','model','during',
                'while','there','patient','group','control',
                'figure','table','authors','being','through',
                'without','showed','where','those','higher',
                'lower','increase','decrease','levels',
                'reduced','increased','observed','treatment',
                'significantly','assessed','could','would',
                'should','investigated','reported','suggested',
                'measured','concentration','sample','samples',
                'plasma','serum','blood','urine','cerebrospinal',
                'usepackage','information','performance'
            )
            GROUP BY word
            ORDER BY freq DESC
            LIMIT 40
        """)
        keywords = [
            {"word": row[0], "freq": row[1]}
            for row in cur.fetchall()
        ]
    return {"keywords": keywords}


@router.get("/status")
def get_server_status(conn: connection = Depends(get_db)):
    """
    Returns the health status of the backend, database, and LLM model.
    """
    import ollama as ollama_client

    try:
        models = ollama_client.list()
        ollama_model = any(
            "llama3" in m["name"] for m in models.get("models", [])
        )
    except Exception:
        ollama_model = False

    return {
        "fastapi_backend": True,
        "database_connection": True,
        "ollama_model": ollama_model,
    }