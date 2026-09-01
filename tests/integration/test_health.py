"""Integration tests for the health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.config import Settings


def test_health_reports_service_metadata(client: TestClient, settings: Settings) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    }


def test_readiness_passes_when_dependencies_are_available(client: TestClient) -> None:
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert [check["name"] for check in body["checks"]] == ["configuration"]


def test_readiness_returns_503_when_a_check_fails(
    client: TestClient,
    monkeypatch: object,
) -> None:
    """A failing dependency must surface as 503, not a 200 with ready=false."""
    from backend.app.api.v1.endpoints import health as health_module
    from backend.app.schemas.common import ReadinessCheck

    monkeypatch.setattr(  # type: ignore[attr-defined]
        health_module,
        "_run_checks",
        lambda: [ReadinessCheck(name="configuration", ready=False, detail="unset")],
    )

    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/does-not-exist").status_code == 404
