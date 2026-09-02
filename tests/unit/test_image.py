"""Unit tests for image decoding and validation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.app.core.exceptions import InvalidImageError, PayloadTooLargeError
from backend.app.utils.image import decode_image

LIMITS = {"max_bytes": 5 * 1024 * 1024, "max_pixels": 40_000_000, "min_edge": 32}


def encode_png(width: int, height: int, colour: tuple[int, int, int] = (0, 0, 255)) -> bytes:
    """Return PNG bytes of a solid-colour image, described in BGR."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return bytes(buffer)


def test_decodes_a_valid_image_to_rgb() -> None:
    # Pure blue in BGR should come back as pure blue in RGB's third channel.
    result = decode_image(encode_png(64, 48, colour=(255, 0, 0)), **LIMITS)

    assert result.shape == (48, 64, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[0, 0], [0, 0, 255])


def test_rejects_empty_payload() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"", **LIMITS)


def test_rejects_undecodable_bytes() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"this is definitely not an image", **LIMITS)


def test_rejects_payload_over_the_byte_limit() -> None:
    with pytest.raises(PayloadTooLargeError):
        decode_image(encode_png(64, 64), **{**LIMITS, "max_bytes": 10})


def test_byte_limit_is_checked_before_decoding() -> None:
    """Oversized junk must fail on size, not reach the decoder."""
    with pytest.raises(PayloadTooLargeError):
        decode_image(b"x" * 100, **{**LIMITS, "max_bytes": 10})


def test_rejects_image_over_the_pixel_limit() -> None:
    with pytest.raises(PayloadTooLargeError):
        decode_image(encode_png(64, 64), **{**LIMITS, "max_pixels": 100})


def test_rejects_image_below_the_minimum_edge() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(encode_png(16, 64), **LIMITS)


def test_accepts_jpeg_as_well_as_png() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok

    assert decode_image(bytes(buffer), **LIMITS).shape == (64, 64, 3)
