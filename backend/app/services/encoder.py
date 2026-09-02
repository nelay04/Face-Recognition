"""Face detection and embedding.

Wraps ``face_recognition`` so the rest of the application never imports dlib
directly, and can be swapped for another backend without touching callers.

Every method here is CPU-bound and blocking. Callers running inside an event
loop must offload them (a sync FastAPI route, or ``asyncio.to_thread``);
calling them from ``async def`` will stall the loop for every request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import face_recognition
import numpy as np
from numpy.typing import NDArray

from backend.app.core.config import DetectionModel
from backend.app.core.exceptions import MultipleFacesError, NoFaceDetectedError
from backend.app.utils.image import RgbImage

logger = logging.getLogger(__name__)

Embedding = NDArray[np.float64]
"""A 128-dimension face descriptor produced by dlib."""

EMBEDDING_DIM = 128


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel coordinates of a detected face, in the source image's scale."""

    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A face found in an image, with its descriptor."""

    box: BoundingBox
    embedding: Embedding


class FaceEncoder:
    """Detects faces and computes their embeddings.

    Stateless apart from configuration, so a single instance is safe to share
    across threads.
    """

    def __init__(
        self,
        *,
        model: DetectionModel = "hog",
        upsample: int = 1,
    ) -> None:
        self._model = model
        self._upsample = upsample

    def detect(self, image: RgbImage) -> list[BoundingBox]:
        """Locate every face in ``image``."""
        locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=self._upsample,
            model=self._model,
        )
        return [BoundingBox(top=t, right=r, bottom=b, left=lft) for t, r, b, lft in locations]

    def encode(self, image: RgbImage) -> list[DetectedFace]:
        """Detect and describe every face in ``image``.

        Returns an empty list when the image contains no faces; that is a
        normal outcome for recognition, so it is not treated as an error here.
        """
        locations = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=self._upsample,
            model=self._model,
        )
        if not locations:
            return []

        embeddings = face_recognition.face_encodings(image, known_face_locations=locations)

        return [
            DetectedFace(
                box=BoundingBox(top=t, right=r, bottom=b, left=lft),
                embedding=np.asarray(embedding, dtype=np.float64),
            )
            for (t, r, b, lft), embedding in zip(locations, embeddings, strict=True)
        ]

    def encode_single(self, image: RgbImage) -> DetectedFace:
        """Describe the one face in ``image``, for enrolment.

        Enrolment is deliberately strict: a photo with two people is ambiguous
        about who is being enrolled, so it is rejected rather than guessed at.

        Raises:
            NoFaceDetectedError: the image contains no face.
            MultipleFacesError: the image contains more than one face.
        """
        faces = self.encode(image)

        if not faces:
            raise NoFaceDetectedError()
        if len(faces) > 1:
            raise MultipleFacesError(f"Found {len(faces)} faces; enrolment requires exactly one.")

        return faces[0]
