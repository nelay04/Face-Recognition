"""Unit tests for settings parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


def test_cors_origins_accepts_comma_separated_string() -> None:
    settings = _settings(cors_origins="http://a.test, http://b.test")

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_parses_from_a_real_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: passing a value as an init kwarg (as above) skips
    pydantic-settings' own env/dotenv parsing entirely, which is a different
    code path from what a real deployment uses. This construction — reading
    an actual .env file with no init kwargs — is what caught a startup crash
    that every other test here missed: pydantic-settings tried to JSON-decode
    the comma-separated string before `_split_origins` ever ran, and blew up
    on a value as ordinary as ``CORS_ORIGINS=http://a,http://b``.
    """
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CORS_ORIGINS=http://a.test,http://b.test\n")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

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
