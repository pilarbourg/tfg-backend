import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag.generation import get_atlas_stream, get_atlas_stream_atlas
from rag.retrieval import get_context
from api.routes import router as dashboard_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    def generate_stream():
        try:
            chunks, context_string = get_context(request.query)
            
            sources = [{"title": c['meta']['title'], "url": c['meta']['url']} for c in chunks]
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            for chunk in get_atlas_stream(request.query, context_string):
                try:
                    content = chunk['response']
                except (TypeError, KeyError):
                    content = chunk[1] if isinstance(chunk, (tuple, list)) else ""

                if content:
                    yield json.dumps({"type": "text", "data": content}) + "\n"
                    
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")

@app.post("/atlas-describe")
async def atlas_describe(request: ChatRequest):
    def generate_stream():
        try:
            chunks, context_string = get_context(request.query)
            
            sources = [{"title": c['meta']['title'], "url": c['meta']['url']} for c in chunks]
            yield json.dumps({"type": "sources", "data": sources}) + "\n"

            for chunk in get_atlas_stream_atlas(request.query, context_string):
                try:
                    content = chunk['response']
                except (TypeError, KeyError):
                    content = chunk[1] if isinstance(chunk, (tuple, list)) else ""

                if content:
                    yield json.dumps({"type": "text", "data": content}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generate_stream(), media_type="application/x-ndjson")