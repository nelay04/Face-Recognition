"""Unit tests for the face encoder.

These drive real dlib. No photograph ships with the repository, so the tests
assert on behaviour that holds for any input — shapes, empty results, and the
enrolment guards — rather than on recognising a specific person.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from backend.app.core.exceptions import MultipleFacesError, NoFaceDetectedError
from backend.app.services.encoder import (
    EMBEDDING_DIM,
    BoundingBox,
    DetectedFace,
    FaceEncoder,
)


@pytest.fixture(scope="module")
def encoder() -> FaceEncoder:
    return FaceEncoder(model="hog", upsample=0)


def blank_image(size: int = 128) -> np.ndarray:
    """A featureless image, which contains no detectable face."""
    return np.zeros((size, size, 3), dtype=np.uint8)


def test_detect_finds_nothing_in_a_blank_image(encoder: FaceEncoder) -> None:
    assert encoder.detect(blank_image()) == []


def test_encode_returns_empty_for_a_faceless_image(encoder: FaceEncoder) -> None:
    """No face is a normal recognition outcome, not an error."""
    assert encoder.encode(blank_image()) == []


def test_encode_single_rejects_an_image_with_no_face(encoder: FaceEncoder) -> None:
    with pytest.raises(NoFaceDetectedError):
        encoder.encode_single(blank_image())


def test_encode_single_rejects_multiple_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enrolment must refuse to guess which of several people to store."""
    encoder = FaceEncoder()
    two_faces = [
        DetectedFace(
            box=BoundingBox(top=0, right=10, bottom=10, left=0),
            embedding=np.zeros(EMBEDDING_DIM),
        )
    ] * 2
    monkeypatch.setattr(encoder, "encode", lambda _: two_faces)

    with pytest.raises(MultipleFacesError):
        encoder.encode_single(blank_image())


def test_encode_single_returns_the_only_face(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = FaceEncoder()
    only = DetectedFace(
        box=BoundingBox(top=1, right=2, bottom=3, left=4),
        embedding=np.zeros(EMBEDDING_DIM),
    )
    monkeypatch.setattr(encoder, "encode", lambda _: [only])

    assert encoder.encode_single(blank_image()) is only


def test_bounding_box_geometry() -> None:
    box = BoundingBox(top=10, right=60, bottom=70, left=20)

    assert box.width == 40
    assert box.height == 60
    assert box.area == 2400


def test_detected_face_is_immutable() -> None:
    """Results are frozen so callers cannot mutate a shared detection."""
    face = DetectedFace(
        box=BoundingBox(top=0, right=1, bottom=1, left=0),
        embedding=np.zeros(EMBEDDING_DIM),
    )

    with pytest.raises(FrozenInstanceError):
        face.box = BoundingBox(top=9, right=9, bottom=9, left=9)  # type: ignore[misc]
