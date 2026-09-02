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
from backend.app.core.device import resolve as resolve_device
from backend.app.core.exceptions import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.services.encoder import FaceEncoder
from backend.app.services.gallery import FaceGallery
from backend.app.services.recognizer import Recognizer

logger = logging.getLogger(__name__)

API_V1_PREFIX = "/api/v1"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown work once per process.

    The encoder loads ~100 MB of model weights, so it is built here and shared
    across requests rather than constructed per call.
    """
    settings: Settings = app.state.settings
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.version, settings.environment)

    # Resolved once at startup, not per request: probing CUDA is cheap but the
    # answer cannot change while the process runs, and the log line below is
    # the only place anyone finds out where detection is actually running.
    device = resolve_device(settings.compute_device, override=settings.detection_model)
    logger.info("Detection running on %s", device)
    app.state.device = device

    app.state.encoder = FaceEncoder(
        model=device.detection_model,
        upsample=settings.detection_upsample,
        max_edge=settings.detection_max_edge,
        encoding_model=settings.encoding_model,
        jitters=settings.encoding_jitters,
        enrolment_jitters=settings.enrolment_jitters,
    )
    app.state.gallery = FaceGallery(settings.gallery_db_path)
    app.state.recognizer = Recognizer(
        app.state.gallery,
        tolerance=settings.match_tolerance,
    )
    logger.info("Loaded recognition services (%d enrolled)", app.state.gallery.count())

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
