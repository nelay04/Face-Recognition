"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Deterministic settings, independent of any local .env."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_name="Face Recognition API",
        version="0.1.0",
        environment="development",
        log_level="WARNING",
        cors_origins=["http://testserver"],
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A test client bound to an app built from the fixture settings."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
