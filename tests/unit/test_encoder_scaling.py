"""Unit tests for detection downscaling.

Detection runs on a shrunk copy for speed, but every coordinate the encoder
returns must be in the caller's original pixel space.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.services.encoder import FaceEncoder, _to_box


def image(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_large_image_is_shrunk_to_the_max_edge() -> None:
    encoder = FaceEncoder(max_edge=640)

    working, scale = encoder._downscale(image(4000, 3000))

    assert max(working.shape[:2]) == 640
    assert scale == pytest.approx(640 / 4000)


def test_aspect_ratio_is_preserved() -> None:
    encoder = FaceEncoder(max_edge=640)

    working, _ = encoder._downscale(image(4000, 2000))

    height, width = working.shape[:2]
    assert width / height == pytest.approx(2.0, rel=0.01)


def test_small_image_is_left_alone() -> None:
    """Shrinking a small frame would destroy the faces in it."""
    encoder = FaceEncoder(max_edge=640)
    original = image(320, 240)

    working, scale = encoder._downscale(original)

    assert scale == 1.0
    assert working.shape == original.shape


def test_image_exactly_at_the_limit_is_untouched() -> None:
    encoder = FaceEncoder(max_edge=640)

    working, scale = encoder._downscale(image(640, 480))

    assert scale == 1.0
    assert working.shape[:2] == (480, 640)


def test_zero_max_edge_disables_downscaling() -> None:
    encoder = FaceEncoder(max_edge=0)

    working, scale = encoder._downscale(image(4000, 3000))

    assert scale == 1.0
    assert working.shape[:2] == (3000, 4000)


def test_tall_images_are_capped_on_their_long_edge() -> None:
    encoder = FaceEncoder(max_edge=640)

    working, _ = encoder._downscale(image(1000, 4000))

    assert working.shape[0] == 640


def test_boxes_are_returned_in_source_coordinates() -> None:
    """A box found at half scale must be doubled back to source pixels."""
    box = _to_box((100, 200, 300, 50), scale=0.5)

    assert (box.top, box.right, box.bottom, box.left) == (200, 400, 600, 100)


def test_boxes_pass_through_unchanged_at_full_scale() -> None:
    box = _to_box((10, 20, 30, 5), scale=1.0)

    assert (box.top, box.right, box.bottom, box.left) == (10, 20, 30, 5)


def test_detect_reports_no_faces_in_a_large_blank_image() -> None:
    """Exercises the real downscale path end to end."""
    encoder = FaceEncoder(max_edge=320, upsample=0)

    assert encoder.detect(image(1600, 1200)) == []
