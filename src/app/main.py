from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.library import router as library_router
from app.api.routes.auth import router as auth_router

app = FastAPI(
    title="PD Metabolomics Knowledge Atlas",
    description="A RAG-powered API for Parkinson's Disease metabolomics research.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(library_router, prefix="/api")
app.include_router(auth_router, prefix="/api")