"""Unit tests for the SQLite face gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.core.exceptions import IdentityExistsError, IdentityNotFoundError
from backend.app.services.encoder import EMBEDDING_DIM
from backend.app.services.gallery import FaceGallery


@pytest.fixture
def gallery(tmp_path: Path) -> FaceGallery:
    return FaceGallery(tmp_path / "gallery.db")


def embedding(seed: int) -> np.ndarray:
    return np.full(EMBEDDING_DIM, float(seed), dtype=np.float64)


def test_add_then_get_round_trips_the_embedding(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding(1))

    stored = gallery.get("nelay")

    assert stored.name == "nelay"
    np.testing.assert_array_equal(stored.embedding, embedding(1))
    assert stored.embedding.dtype == np.float64


def test_add_rejects_a_duplicate_name(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding(1))

    with pytest.raises(IdentityExistsError):
        gallery.add("nelay", embedding(2))


def test_add_rejects_a_wrongly_shaped_embedding(gallery: FaceGallery) -> None:
    with pytest.raises(ValueError, match="shape"):
        gallery.add("nelay", np.zeros(5, dtype=np.float64))


def test_get_raises_for_an_unknown_name(gallery: FaceGallery) -> None:
    with pytest.raises(IdentityNotFoundError):
        gallery.get("nobody")


def test_delete_removes_the_identity(gallery: FaceGallery) -> None:
    gallery.add("nelay", embedding(1))

    gallery.delete("nelay")

    assert gallery.count() == 0
    with pytest.raises(IdentityNotFoundError):
        gallery.get("nelay")


def test_delete_raises_for_an_unknown_name(gallery: FaceGallery) -> None:
    with pytest.raises(IdentityNotFoundError):
        gallery.delete("nobody")


def test_list_is_empty_for_a_fresh_gallery(gallery: FaceGallery) -> None:
    assert gallery.list_all() == []
    assert gallery.count() == 0


def test_as_matrix_stacks_every_embedding(gallery: FaceGallery) -> None:
    gallery.add("a", embedding(1))
    gallery.add("b", embedding(2))

    names, matrix = gallery.as_matrix()

    assert names == ["a", "b"]
    assert matrix.shape == (2, EMBEDDING_DIM)
    np.testing.assert_array_equal(matrix[1], embedding(2))


def test_as_matrix_keeps_its_shape_when_empty(gallery: FaceGallery) -> None:
    """Callers rely on the second dimension even with nothing enrolled."""
    names, matrix = gallery.as_matrix()

    assert names == []
    assert matrix.shape == (0, EMBEDDING_DIM)


def test_data_survives_reopening_the_database(tmp_path: Path) -> None:
    path = tmp_path / "gallery.db"
    FaceGallery(path).add("nelay", embedding(3))

    reopened = FaceGallery(path)

    np.testing.assert_array_equal(reopened.get("nelay").embedding, embedding(3))


def test_creates_the_parent_directory(tmp_path: Path) -> None:
    gallery = FaceGallery(tmp_path / "nested" / "deeper" / "gallery.db")

    assert gallery.count() == 0
