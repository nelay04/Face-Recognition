"""Unit tests for settings parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def test_cors_origins_accepts_comma_separated_string() -> None:
    settings = _settings(cors_origins="http://a.test, http://b.test")

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_ignores_blank_entries() -> None:
    assert _settings(cors_origins="http://a.test, ,").cors_origins == ["http://a.test"]


def test_is_production_tracks_environment() -> None:
    assert _settings(environment="production").is_production is True
    assert _settings(environment="development").is_production is False


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings(environment="staging-ish")


def test_port_must_be_a_valid_tcp_port() -> None:
    with pytest.raises(ValidationError):
        _settings(port=70_000)
