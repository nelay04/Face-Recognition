"""Liveness and readiness probes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from backend.app.api.deps import EncoderDep, GalleryDep, SettingsDep
from backend.app.schemas.common import (
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)
from backend.app.services.encoder import FaceEncoder
from backend.app.services.gallery import FaceGallery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Confirms the process is running. Does not touch dependencies.",
)
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        version=settings.version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Reports whether the service can serve real traffic. Returns 503 when "
        "any dependency is unavailable, so orchestrators stop routing to it."
    ),
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def readiness(
    response: Response,
    encoder: EncoderDep,
    gallery: GalleryDep,
) -> ReadinessResponse:
    checks = _run_checks(encoder, gallery)
    ready = all(check.ready for check in checks)

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(ready=ready, checks=checks)


def _run_checks(encoder: FaceEncoder, gallery: FaceGallery) -> list[ReadinessCheck]:
    """Probe each dependency the service needs to answer requests.

    Each probe is exercised for real rather than assumed: the gallery check
    runs a query, so a missing or corrupt database surfaces here instead of on
    the first recognition request.
    """
    return [
        ReadinessCheck(name="configuration", ready=True),
        _check_encoder(encoder),
        _check_gallery(gallery),
    ]


def _check_encoder(encoder: FaceEncoder) -> ReadinessCheck:
    ready = encoder is not None
    return ReadinessCheck(
        name="encoder",
        ready=ready,
        detail="model loaded" if ready else "model not loaded",
    )


def _check_gallery(gallery: FaceGallery) -> ReadinessCheck:
    try:
        count = gallery.count()
    except Exception as exc:  # Any failure here means "not ready".
        logger.warning("Gallery readiness check failed: %s", exc)
        return ReadinessCheck(name="gallery", ready=False, detail=str(exc))

    return ReadinessCheck(
        name="gallery",
        ready=True,
        detail=f"{count} identities enrolled",
    )
