"""Aggregates the v1 endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.v1.endpoints import faces, health, recognize

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(faces.router)
api_router.include_router(recognize.router)
