"""Matching probe embeddings against the enrolled gallery."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backend.app.services.encoder import BoundingBox, DetectedFace, Embedding
from backend.app.services.gallery import FaceGallery

logger = logging.getLogger(__name__)

UNKNOWN = "Unknown"
"""Label for a face that matched nobody within tolerance."""


@dataclass(frozen=True, slots=True)
class Recognition:
    """One detected face, resolved to an identity or to ``Unknown``."""

    box: BoundingBox
    name: str
    distance: float
    confidence: float

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN


class Recognizer:
    """Resolves face embeddings to enrolled identities.

    Holds no state of its own: the gallery is read on each call, so enrolments
    take effect immediately without a cache to invalidate.
    """

    def __init__(self, gallery: FaceGallery, *, tolerance: float = 0.6) -> None:
        self._gallery = gallery
        self._tolerance = tolerance

    def identify(self, faces: list[DetectedFace]) -> list[Recognition]:
        """Label each detected face.

        The gallery is loaded once for the whole batch rather than per face,
        which matters when a frame contains several people.
        """
        if not faces:
            return []

        names, matrix = self._gallery.as_matrix()

        # Nothing enrolled: every face is legitimately unknown, not an error.
        if not names:
            return [_unknown(face.box) for face in faces]

        return [self._match(face, names, matrix) for face in faces]

    def _match(
        self,
        face: DetectedFace,
        names: list[str],
        matrix: np.ndarray,
    ) -> Recognition:
        distances = _euclidean_distances(matrix, face.embedding)
        best = int(np.argmin(distances))
        distance = float(distances[best])

        if distance > self._tolerance:
            return _unknown(face.box, distance=distance)

        return Recognition(
            box=face.box,
            name=names[best],
            distance=distance,
            confidence=_confidence(distance, self._tolerance),
        )


def _euclidean_distances(matrix: np.ndarray, probe: Embedding) -> np.ndarray:
    """Distance from ``probe`` to every row of ``matrix``.

    This is the same metric dlib's own ``compare_faces`` uses, computed for
    the whole gallery in one vectorised pass.
    """
    return np.linalg.norm(matrix - probe, axis=1)


def _confidence(distance: float, tolerance: float) -> float:
    """Map a distance onto a 0..1 score.

    A convenience for display only — it is a linear rescaling of distance
    within the tolerance band, not a calibrated probability, so it should not
    be used to make decisions. The threshold comparison is what decides a
    match.
    """
    if tolerance <= 0:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - distance / tolerance)), 4)


def _unknown(box: BoundingBox, *, distance: float = float("inf")) -> Recognition:
    """An unmatched face. Confidence is zero by construction."""
    return Recognition(box=box, name=UNKNOWN, distance=distance, confidence=0.0)
