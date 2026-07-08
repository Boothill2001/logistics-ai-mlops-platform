"""FastAPI app factory."""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app

from src.api.middleware import RequestContextMiddleware
from src.api.routes_predict import router as predict_router

logging.basicConfig(level=logging.INFO, format="%(message)s")

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="Logistics AI MLOps Platform", version="1.0.0")
    app.add_middleware(RequestContextMiddleware)
    app.include_router(predict_router)
    try:
        from src.api.routes_copilot import router as copilot_router
        app.include_router(copilot_router)
    except ImportError as exc:  # pragma: no cover
        logging.getLogger("api").warning("copilot disabled: %s", exc)
    app.mount("/metrics", make_asgi_app())

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
