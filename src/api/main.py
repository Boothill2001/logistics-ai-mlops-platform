"""FastAPI app factory."""
import logging

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from src.api.middleware import RequestContextMiddleware
from src.api.routes_predict import router as predict_router

logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app() -> FastAPI:
    app = FastAPI(title="Logistics AI MLOps Platform", version="1.0.0")
    app.add_middleware(RequestContextMiddleware)
    app.include_router(predict_router)
    # Copilot routes are registered lazily so the ML API works even if
    # copilot deps (chroma/langgraph) are unavailable — graceful degradation.
    try:
        from src.api.routes_copilot import router as copilot_router
        app.include_router(copilot_router)
    except ImportError as exc:  # pragma: no cover
        logging.getLogger("api").warning("copilot disabled: %s", exc)
    app.mount("/metrics", make_asgi_app())
    return app


app = create_app()
