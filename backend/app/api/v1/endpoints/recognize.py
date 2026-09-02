"""Recognition: identify faces in a submitted frame."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from backend.app.api.deps import EncoderDep, RecognizerDep, SettingsDep
from backend.app.api.uploads import read_image_upload
from backend.app.schemas.common import ErrorResponse
from backend.app.schemas.recognition import RecognitionResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recognition"])


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Identify faces in an image",
    description=(
        "Detects every face in the image and matches each against the enrolled "
        "gallery. Faces beyond the configured tolerance are returned as "
        "'Unknown'. An image with no faces is a successful, empty result."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
    },
)
async def recognize(
    settings: SettingsDep,
    encoder: EncoderDep,
    recognizer: RecognizerDep,
    image: Annotated[UploadFile, File(description="Frame to identify faces in.")],
) -> RecognitionResponse:
    picture = await read_image_upload(image, settings)

    started = time.perf_counter()

    # Both steps are blocking C++ / numpy work, so they run off the event loop.
    faces = await asyncio.to_thread(encoder.encode, picture)
    recognitions = await asyncio.to_thread(recognizer.identify, faces)

    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.debug("Recognised %d face(s) in %.1f ms", len(recognitions), elapsed_ms)
    return RecognitionResponse.from_domain(recognitions, processing_ms=elapsed_ms)
