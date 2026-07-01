# Parkinson's Metabolic Atlas — Backend

The retrieval-augmented generation (RAG) pipeline for the Parkinson's Metabolic Atlas. This service ingests Parkinson's Disease (PD) metabolomics literature, indexes it as vector embeddings in a PostgreSQL database, and exposes a REST API that returns evidence-grounded responses to natural language queries.

This repository contains the backend only. The React client is maintained in a separate repository (see [Related Repositories](#related-repositories)).

## Overview

The backend implements an Extract-Transform-Load (ETL) workflow and a two-stage retrieval architecture:

- **Extract** — Retrieves open-access full-text PDFs from PubMed Central via the PMC Open Access Web Service API, with fallbacks to Unpaywall.
- **Transform** — Parses PDFs into structured Markdown using Docling, then applies a noise-removal filter to discard bibliographies, footnotes, and other non-content text.
- **Load** — Splits text into overlapping chunks, embeds them with a PubMedBERT model, and stores the vectors, chunk text, and source metadata in PostgreSQL using the `pgvector` extension.
- **Retrieve** — Combines semantic (vector) search and PostgreSQL full-text keyword search, then re-ranks the merged candidates with a FlashRank cross-encoder.
- **Generate** — Passes the top-ranked chunks and the user query to a locally hosted Qwen 2.5 model (via Ollama) under a prompt that grounds claims in the retrieved context.

## Getting Started

Clone the repository and install dependencies:

```bash
git clone https://github.com/pilarbourg/tfg-backend.git
cd tfg-backend
pip install -r requirements.txt
```

Run the API server:

```bash
PYTHONPATH=src/app python -m uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`, matching the `VITE_API_BASE_URL` expected by the frontend.

## Configuration

The backend is configured through environment variables. Create a `.env` file at the project root with the following variables:

```
ADMIN_USERNAME
ADMIN_PASSWORD
UNPAYWALL_EMAIL
DB_NAME
DB_HOST
DB_USER
DB_PASSWORD
DB_SSLMODE
GROQ_API_KEY
```

## ETL Pipeline

The ingestion pipeline can be run to populate or update the knowledge base:

```bash
python app/core/etl/full_pipeline.py
```

Ingestion is automated via GitHub Actions, which processes up to 10 new papers each Monday: extracting text, chunking, embedding, and loading the results into the hosted database.

```
.github/workflows/ingest.yml
```

## Database Schema

The PostgreSQL database contains five tables:

- `research_papers` — Dynamic knowledge base at 512 tokens: metadata, chunk text, and vector embeddings.
- `limited_papers` — Dynamic knowledge base at 336 tokens: metadata, chunk text, and vector embeddings.
- `validation_papers` — Static validation set (10 PMIDs) at 512 tokens used for evaluation. Shares the same columns as `research_papers`; no relationship exists between the two.
- `limited_validation_papers` — Static validation set (10 PMIDs) at 336 tokens used for evaluation. Shares the same columns as `research_papers`; no relationship exists between the two.
- `ingestion_state` — Single `updated_at` timestamp marking the most recent ingestion, used to avoid duplicate processing.

## API Endpoints

| Method | Route | Purpose |
| ------ | ----- | ------- |
| POST | `/api/chat` | Streams RAG-generated responses to research queries as newline-delimited JSON. |
| POST | `/api/atlas-describe` | Streams neuroanatomical metabolite descriptions for the 3D atlas. |
| POST | `/api/auth/login` | Validates administrative credentials against environment variables. |
| GET | `/api/dashboard/stats` | Returns counts of papers, chunks, and full-text entries. |
| GET | `/api/dashboard/keywords` | Returns the most frequent keywords in the knowledge base. |
| GET | `/api/search` | Returns the top-10 papers matching a keyword via full-text search. |

## Project Structure

```
root/
  src/
    app/
      api/                         FastAPI routers and endpoint handlers
      etl/                         Extract, transform, and load modules
      retrieval/                   Hybrid search and FlashRank re-ranking
      generation/                  Prompt construction and Ollama integration
      db/                          Database models and connection
      main.py
    data/
      metadata_index.json          Metadata of research papers found in db
      metadata_validation.json     Metadata of validation papers found in db
    downloads/                     Temporarily contains research paper PDFs as they are processed
    results/                       Contains research papers in markdown

  test/
    unit/                          Unit tests
    integration/                   Postman integration testing
    ragas/                         Ragas evaluation and validation
    synonyms/                      Metabolic synonyms validation
    validation_set.json            PMIDs of papers found in validation set

  requirements.txt

```

## Related Repositories

- **Frontend (React client)** — https://github.com/pilarbourg/tfg-frontend.git

## Acknowledgements

Developed as part of the Trabajo Fin de Grado (TFG) at CEU San Pablo University.