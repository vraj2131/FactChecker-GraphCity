"""
Phase 11: FastAPI application entry point.

Registers:
- CORS middleware (allows React dev server on localhost:5173)
- /api/v1/verify-claim route
- /health route
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_sources import router as sources_router
from backend.app.api.routes_verify import router as verify_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FactGraph City API",
    description="Fact-checking pipeline — claim → graph JSON",
    version="0.11.0",
)

# -------------------------------------------------------------------
# CORS — allow the React frontend (Vite dev server + production build)
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA / alternative dev port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
app.include_router(health_router)
app.include_router(verify_router)
app.include_router(sources_router)
