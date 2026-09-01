"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from backend.app.api.deps import SettingsDep
from backend.app.schemas.common import (
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)

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
async def readiness(response: Response) -> ReadinessResponse:
    checks = _run_checks()
    ready = all(check.ready for check in checks)

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(ready=ready, checks=checks)


def _run_checks() -> list[ReadinessCheck]:
    """Probe each dependency the service needs to answer requests.

    Only configuration is checked today. Recognition-model loading and gallery
    availability join this list when those features land.
    """
    return [ReadinessCheck(name="configuration", ready=True)]
