"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import Settings
from backend.app.main import create_app

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Deterministic settings with a gallery isolated per test."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_name="Face Recognition API",
        version="0.1.0",
        environment="development",
        log_level="WARNING",
        cors_origins=["http://testserver"],
        gallery_db_path=tmp_path / "gallery.db",
        # Keep detection cheap; tests assert on plumbing, not accuracy.
        detection_upsample=0,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A test client bound to an app built from the fixture settings."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def blank_png() -> bytes:
    """A valid image that contains no face."""
    return encode_png(np.zeros((128, 128, 3), dtype=np.uint8))


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return bytes(buffer)
