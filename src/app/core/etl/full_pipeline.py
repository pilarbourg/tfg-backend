import logging
import os
import json
from psycopg2.extensions import connection
from app.services.db.connection import get_db_connection
from app.core.etl.pipeline.extract_pdf import download_pmc_pdf
from app.core.etl.pipeline.transform_pdf import process_pdf
from app.core.etl.pipeline.load_pdf import ingest_paper
from app.core.schemas.research_paper import ResearchPaper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def run_etl_pipeline(paper: ResearchPaper, conn: connection) -> None:
    """
    Runs entire ETL pipeline.

    Parameters
    ----------
    paper : ResearchPaper
        Research article metadata including title, abstract, pmid
    """
    if os.path.exists(f"results/PMC{paper.pmcid}.md"):
      logging.info(f"PMC{paper.pmcid} already processed, skipping.")
      return

    if not download_pmc_pdf(paper.pmcid, paper.doi):
      logging.error(f"Download failed for PMC{paper.pmcid}, skipping.")
      return
    
    md_path = process_pdf(paper.pmcid)

    if md_path is None:
      return 

    ingest_paper(
      title=paper.title,
      pmid=paper.pmid,
      pmcid=paper.pmcid,
      source_id=paper.doi or paper.pmid,
      abstract=paper.abstract,
      md_path=md_path,
      conn=conn
    )

    logging.info("Successfully ran ETL pipeline.")

def run_pipeline() -> None:
    """
    Orchestrates the full ETL pipeline across all papers in the metadata index.
    Opens a single database connection, filters papers with full text available,
    and runs the extract, transform, and load steps for each one.
    """
        
    conn = get_db_connection()
    try:
      with open("data/metadata_index.json", "r") as f:
        metadata = json.load(f)

      for entry in metadata:
          if entry.get("has_full_text", False):
              paper = ResearchPaper(**entry)
              run_etl_pipeline(paper, conn)
    finally:
        conn.close()

if __name__ == "__main__":
    run_pipeline()