"""Unit tests for gallery matching.

These use synthetic embeddings rather than real photographs: the matching
logic is pure geometry, so it can be tested exactly without invoking dlib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.services.encoder import EMBEDDING_DIM, BoundingBox, DetectedFace
from backend.app.services.gallery import FaceGallery
from backend.app.services.recognizer import UNKNOWN, Recognizer

BOX = BoundingBox(top=10, right=60, bottom=70, left=20)


@pytest.fixture
def gallery(tmp_path: Path) -> FaceGallery:
    return FaceGallery(tmp_path / "gallery.db")


def embedding(*, offset: float = 0.0) -> np.ndarray:
    """An embedding a known distance from the origin vector.

    Only the first component varies, so the Euclidean distance from the zero
    vector is exactly ``offset``.
    """
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    vector[0] = offset
    return vector


def face(*, offset: float = 0.0) -> DetectedFace:
    return DetectedFace(box=BOX, embedding=embedding(offset=offset))


def test_identifies_an_enrolled_face(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding())
    recognizer = Recognizer(gallery, tolerance=0.6)

    [result] = recognizer.identify([face(offset=0.1)])

    assert result.name == "nelay"
    assert result.is_known
    assert result.distance == pytest.approx(0.1)


def test_labels_a_distant_face_unknown(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding())
    recognizer = Recognizer(gallery, tolerance=0.6)

    [result] = recognizer.identify([face(offset=0.9)])

    assert result.name == UNKNOWN
    assert not result.is_known
    assert result.confidence == 0.0


def test_picks_the_nearest_of_several_identities(gallery: FaceGallery) -> None:
    gallery.add("far", embedding(offset=0.5))
    gallery.add("near", embedding(offset=0.12))
    recognizer = Recognizer(gallery, tolerance=0.6)

    [result] = recognizer.identify([face(offset=0.1)])

    assert result.name == "near"


def test_everything_is_unknown_when_nothing_is_enrolled(gallery: FaceGallery) -> None:
    recognizer = Recognizer(gallery, tolerance=0.6)

    [result] = recognizer.identify([face()])

    assert result.name == UNKNOWN
    assert result.box == BOX


def test_returns_nothing_for_no_faces(gallery: FaceGallery) -> None:
    assert Recognizer(gallery).identify([]) == []


def test_handles_several_faces_in_one_frame(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding())
    recognizer = Recognizer(gallery, tolerance=0.6)

    results = recognizer.identify([face(offset=0.05), face(offset=2.0)])

    assert [result.name for result in results] == ["nelay", UNKNOWN]


def test_tolerance_is_the_deciding_boundary(gallery: FaceGallery) -> None:
    """The same probe flips identity purely on the configured tolerance."""
    gallery.add("nelay", embedding())

    strict = Recognizer(gallery, tolerance=0.3).identify([face(offset=0.4)])
    lenient = Recognizer(gallery, tolerance=0.5).identify([face(offset=0.4)])

    assert strict[0].name == UNKNOWN
    assert lenient[0].name == "nelay"


def test_confidence_falls_as_distance_grows(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding())
    recognizer = Recognizer(gallery, tolerance=0.6)

    close = recognizer.identify([face(offset=0.05)])[0]
    distant = recognizer.identify([face(offset=0.5)])[0]

    assert close.confidence > distant.confidence
    assert 0.0 <= distant.confidence <= 1.0


def test_an_exact_match_scores_full_confidence(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding())

    [result] = Recognizer(gallery, tolerance=0.6).identify([face()])

    assert result.distance == pytest.approx(0.0)
    assert result.confidence == pytest.approx(1.0)


def test_newly_enrolled_identities_are_visible_immediately(gallery: FaceGallery) -> None:
    """The recogniser must not cache the gallery between calls."""
    recognizer = Recognizer(gallery, tolerance=0.6)
    assert recognizer.identify([face()])[0].name == UNKNOWN

    gallery.add("nelay", embedding())

    assert recognizer.identify([face()])[0].name == "nelay"
