"""
Health check routes.

Endpoints:
  GET /health       — basic liveness probe (always returns 200 if the server is up)
  GET /health/ready — readiness probe (checks models are loaded and ready)
"""

import logging

from fastapi import APIRouter

from backend.app.dependencies import get_verify_service, get_graph_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
def health_live():
    """Returns 200 if the server process is running."""
    return {"status": "ok"}


@router.get("/health/ready", summary="Readiness probe")
def health_ready():
    """
    Returns 200 if all services are initialised and ready to serve requests.
    Triggers lazy singleton construction on first call.
    """
    try:
        get_verify_service()
        get_graph_service()
        return {"status": "ready"}
    except Exception as exc:
        logger.exception("Readiness check failed")
        return {"status": "not_ready", "detail": str(exc)}
