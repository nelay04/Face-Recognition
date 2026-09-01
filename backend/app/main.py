"""ASGI entrypoint: builds the FastAPI application."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.v1.router import api_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import configure_logging

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown work once per process."""
    settings: Settings = app.state.settings
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.version, settings.environment)
    yield
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Written as a factory so tests can construct an isolated instance with
    their own settings rather than importing a module-level singleton.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        # Hide interactive docs outside development.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=API_V1_PREFIX)
    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the static UI, if it is present alongside the backend."""
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        logger.warning("Frontend not found at %s; serving API only", FRONTEND_DIR)
        return

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index_page() -> FileResponse:
        return FileResponse(index)


app = create_app()
