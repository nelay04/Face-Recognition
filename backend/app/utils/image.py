"""Image decoding and validation helpers."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.app.core.exceptions import InvalidImageError, PayloadTooLargeError

RgbImage = NDArray[np.uint8]
"""An HxWx3 array of 8-bit RGB pixels — the form dlib expects."""


def decode_image(
    data: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
    min_edge: int,
) -> RgbImage:
    """Decode raw upload bytes into an RGB image array.

    Validation happens in cost order: the byte-length check is free and runs
    before decoding, so an oversized upload never reaches the decoder.

    Raises:
        PayloadTooLargeError: the payload exceeds ``max_bytes``, or decodes to
            more than ``max_pixels`` (a decompression bomb).
        InvalidImageError: the bytes are empty, undecodable, not 3-channel,
            or smaller than ``min_edge`` on either side.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"Image exceeds the {max_bytes} byte limit ({len(data)} bytes).")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("The uploaded file could not be decoded as an image.")

    height, width = image.shape[:2]

    if height * width > max_pixels:
        raise PayloadTooLargeError(
            f"Image resolution {width}x{height} exceeds the {max_pixels} pixel limit."
        )

    if min(height, width) < min_edge:
        raise InvalidImageError(
            f"Image is too small: {width}x{height}, minimum edge is {min_edge}px."
        )

    # IMREAD_COLOR always yields 3 channels, but be explicit rather than trust it.
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("Expected a 3-channel colour image.")

    # OpenCV decodes to BGR; dlib and face_recognition expect RGB.
    # cvtColor is typed loosely upstream; the dtype is unchanged by the convert.
    return cast(RgbImage, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
