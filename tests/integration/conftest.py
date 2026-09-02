"""Fixtures for endpoint tests.

The encoder is stubbed here on purpose. No photograph ships with this
repository, and these tests are about HTTP plumbing — status codes, response
shapes, error mapping. Real dlib behaviour is covered in the unit tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api.deps import get_encoder
from backend.app.services.encoder import (
    EMBEDDING_DIM,
    BoundingBox,
    DetectedFace,
    FaceEncoder,
)

BOX = BoundingBox(top=10, right=60, bottom=70, left=20)


def embedding(offset: float = 0.0) -> np.ndarray:
    """An embedding exactly ``offset`` away from the origin vector."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    vector[0] = offset
    return vector


def detected(offset: float = 0.0) -> DetectedFace:
    return DetectedFace(box=BOX, embedding=embedding(offset))


class StubEncoder:
    """Returns a scripted list of faces, ignoring the image entirely."""

    def __init__(self, faces: list[DetectedFace]) -> None:
        self.faces = faces

    def encode(self, _image: object) -> list[DetectedFace]:
        return list(self.faces)

    def encode_single(self, image: object) -> DetectedFace:
        # Reuse the real guard clauses so their error mapping is exercised.
        return FaceEncoder.encode_single(self, image)  # type: ignore[arg-type]


StubFactory = Callable[[list[DetectedFace]], None]


@pytest.fixture
def stub_encoder(client: TestClient) -> Iterator[StubFactory]:
    """Replace the app's encoder with one returning the given faces."""

    def use(faces: list[DetectedFace]) -> None:
        client.app.dependency_overrides[get_encoder] = lambda: StubEncoder(faces)

    yield use

    client.app.dependency_overrides.clear()
