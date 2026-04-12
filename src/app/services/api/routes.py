from fastapi import APIRouter
import os
from dotenv import load_dotenv
from db.connection import get_db_connection
import ollama as ollama_client
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import secrets

load_dotenv()
router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.get("/api/dashboard/stats")
def get_dashboard_stats():
    conn = get_db_connection()

    try:
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

            # Top journals by paper count (extracted from DOI prefix)
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

    finally:
        conn.close()

@router.get("/api/dashboard/keywords")
def get_keywords():
    conn = get_db_connection()
    try:
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
                    'plasma','serum','blood','urine','cerebrospinal','usepackage', 'information', 'performance'
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
    finally:
        conn.close()

@router.get("/api/dashboard/status")
def get_server_status():
    try:
        conn = get_db_connection()
        conn.close()
        database_connection = True
    except Exception:
        database_connection = False

    try:
        models = ollama_client.list()
        ollama_model = any(
            "llama3" in m["name"] for m in models.get("models", [])
        )
    except Exception:
        ollama_model = False

    return {
        "fastapi_backend": True,
        "database_connection": database_connection,
        "ollama_model": ollama_model,
    }

@router.post("/api/auth/login")
def login(request: LoginRequest):
    expected_user = os.getenv("ADMIN_USERNAME")
    expected_pass = os.getenv("ADMIN_PASSWORD")
    
    print(f"DEBUG: expected_user='{expected_user}' expected_pass='{expected_pass}'")
    print(f"DEBUG: got username='{request.username}' password='{request.password}'")

    if not secrets.compare_digest(request.username, expected_user) or \
       not secrets.compare_digest(request.password, expected_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"success": True}