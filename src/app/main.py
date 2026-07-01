from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.auth import router as auth_router
from app.api.routes.search import router as search_router

app = FastAPI(
    title="PD Metabolomics Knowledge Atlas API",
    description="Backend for Parkinson's Disease metabolomics RAG pipeline.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api", tags=["RAG Chat Engine"])
app.include_router(dashboard_router, prefix="/api", tags=["Dashboard Analytics"])
app.include_router(auth_router, prefix="/api", tags=["Authentication"])
app.include_router(search_router, prefix="/api", tags=["Authentication"])