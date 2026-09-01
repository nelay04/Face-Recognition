"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from backend.app.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Injected application settings.

Declared as a type alias so routes stay readable and tests can override the
underlying ``get_settings`` provider in one place.
"""
