import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from psycopg2.extensions import connection
from pydantic import BaseModel
from app.api.dependencies import get_db
from app.core.etl.obtain_ids import fetch_paper_metadata, get_pmcid_from_pmid, get_pmid_from_pmcid
from app.core.etl.full_pipeline import run_etl_pipeline
from app.core.schemas.research_paper import ResearchPaper

router = APIRouter(prefix="/library", tags=["library"])

@router.get("/search")
def search_metabolite_mentions(conn: connection = Depends(get_db)):
    """
    Returns top-10 articles which mention the given metabolite.
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

        cur.execute("SELECT COUNT(*) FROM research_papers")
        total_chunks = cur.fetchone()[0]

    return {
        "total_papers": total_papers,
        "full_text_count": full_text_count,
        "abstract_only": total_papers - full_text_count,
        "total_chunks": total_chunks,
    }