"""Shared response envelopes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload: the process is up and serving."""

    status: Literal["ok"] = "ok"
    app: str = Field(description="Human-readable service name.")
    version: str = Field(description="Running service version.")
    environment: str = Field(description="Deployment environment.")


class ReadinessCheck(BaseModel):
    """Result of a single dependency probe."""

    name: str
    ready: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness payload: whether the service can handle real requests."""

    ready: bool
    checks: list[ReadinessCheck] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Uniform error body returned by every failing endpoint."""

    code: str = Field(description="Stable, machine-readable error identifier.")
    message: str = Field(description="Human-readable description.")
