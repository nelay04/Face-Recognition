"""Turning multipart uploads into images.

Lives in the API layer because it deals in ``UploadFile``; the services below
it only ever see decoded arrays.
"""

from __future__ import annotations

from fastapi import UploadFile

from backend.app.core.config import Settings
from backend.app.core.exceptions import PayloadTooLargeError
from backend.app.utils.image import RgbImage, decode_image


async def read_image_upload(file: UploadFile, settings: Settings) -> RgbImage:
    """Read and decode an uploaded image.

    Starlette reports the declared size before the body is consumed, so an
    oversized upload is rejected without being read into memory. The decoder
    re-checks the real length, since the declared size is client-supplied and
    may be absent or wrong.
    """
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"Image exceeds the {settings.max_upload_bytes} byte limit ({file.size} bytes)."
        )

    data = await file.read()

    return decode_image(
        data,
        max_bytes=settings.max_upload_bytes,
        max_pixels=settings.max_image_pixels,
        min_edge=settings.min_image_edge,
    )
