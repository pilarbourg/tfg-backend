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


class AddPaperRequest(BaseModel):
    pmcid: str


@router.post("/add")
async def add_paper(request: AddPaperRequest, conn: connection = Depends(get_db)):
    """
    Adds a single paper to the corpus by PMCID.
    Streams pipeline progress back to the client as newline-delimited JSON.
    """
    def generate_stream():
        try:
            yield json.dumps({"type": "status", "data": f"Fetching metadata for PMC{request.pmcid}..."}) + "\n"

            pmid = get_pmid_from_pmcid(request.pmcid)
            if not pmid:
                yield json.dumps({"type": "error", "data": f"Could not find PMID for PMC{request.pmcid}."}) + "\n"
                return

            entry = fetch_paper_metadata(pmid)
            if not entry:
                yield json.dumps({"type": "error", "data": f"Could not fetch metadata for PMID {pmid}."}) + "\n"
                return

            pmcid = get_pmcid_from_pmid(pmid)
            if not pmcid:
                yield json.dumps({"type": "error", "data": f"PMC{request.pmcid} does not have full text available."}) + "\n"
                return

            entry["pmcid"] = pmcid
            entry["has_full_text"] = True

            yield json.dumps({"type": "status", "data": f"Found: {entry['title']}"}) + "\n"
            yield json.dumps({"type": "status", "data": "Downloading PDF..."}) + "\n"

            paper = ResearchPaper(**entry)
            run_etl_pipeline(paper, conn)

            yield json.dumps({"type": "status", "data": "Processing and ingesting paper..."}) + "\n"
            yield json.dumps({"type": "complete", "data": f"PMC{request.pmcid} successfully added to the library."}) + "\n"

        except Exception as e:
            logging.error(f"Error adding paper PMC{request.pmcid}: {e}")
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")


@router.get("/stats")
def get_library_stats(conn: connection = Depends(get_db)):
    """
    Returns high-level statistics about the current paper library.
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