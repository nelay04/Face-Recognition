"""FastAPI dependency providers."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.app.core.config import Settings, get_settings
from backend.app.core.device import DeviceChoice
from backend.app.services.encoder import FaceEncoder
from backend.app.services.gallery import FaceGallery
from backend.app.services.recognizer import Recognizer

SettingsDep = Annotated[Settings, Depends(get_settings)]
"""Injected application settings.

Declared as a type alias so routes stay readable and tests can override the
underlying ``get_settings`` provider in one place.
"""


def get_encoder(request: Request) -> FaceEncoder:
    """The shared encoder, built once during startup.

    Model weights are ~100 MB and slow to load, so they are loaded in the
    lifespan handler and reused rather than constructed per request.
    """
    encoder: FaceEncoder = request.app.state.encoder
    return encoder


def get_device(request: Request) -> DeviceChoice:
    """The device detection was resolved onto at startup.

    Resolved once in the lifespan handler, since CUDA availability cannot
    change while the process runs.
    """
    device: DeviceChoice = request.app.state.device
    return device


def get_gallery(request: Request) -> FaceGallery:
    """The shared identity gallery."""
    gallery: FaceGallery = request.app.state.gallery
    return gallery


def get_recognizer(request: Request) -> Recognizer:
    """The shared recogniser."""
    recognizer: Recognizer = request.app.state.recognizer
    return recognizer


EncoderDep = Annotated[FaceEncoder, Depends(get_encoder)]
DeviceDep = Annotated[DeviceChoice, Depends(get_device)]
GalleryDep = Annotated[FaceGallery, Depends(get_gallery)]
RecognizerDep = Annotated[Recognizer, Depends(get_recognizer)]
