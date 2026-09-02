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
from typing import cast

import cv2
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
        max_edge: int = 640,
    ) -> None:
        self._model = model
        self._upsample = upsample
        self._max_edge = max_edge

    def detect(self, image: RgbImage) -> list[BoundingBox]:
        """Locate every face in ``image``, in source-image coordinates."""
        working, scale = self._downscale(image)
        locations = self._locate(working)
        return [_to_box(location, scale) for location in locations]

    def encode(self, image: RgbImage) -> list[DetectedFace]:
        """Detect and describe every face in ``image``.

        Returns an empty list when the image contains no faces; that is a
        normal outcome for recognition, so it is not treated as an error here.
        """
        working, scale = self._downscale(image)

        locations = self._locate(working)
        if not locations:
            return []

        # Embeddings come from the downscaled frame too, so that enrolment and
        # recognition always describe faces at the same working resolution and
        # their distances stay comparable.
        embeddings = face_recognition.face_encodings(working, known_face_locations=locations)

        return [
            DetectedFace(
                box=_to_box(location, scale),
                embedding=np.asarray(embedding, dtype=np.float64),
            )
            for location, embedding in zip(locations, embeddings, strict=True)
        ]

    def _locate(self, image: RgbImage) -> list[tuple[int, int, int, int]]:
        locations: list[tuple[int, int, int, int]] = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=self._upsample,
            model=self._model,
        )
        return locations

    def _downscale(self, image: RgbImage) -> tuple[RgbImage, float]:
        """Shrink ``image`` so its longest edge is at most ``max_edge``.

        Detection cost grows with pixel count, so a phone photo can take
        seconds while a webcam frame takes milliseconds. Capping the working
        size makes the cost roughly constant regardless of what is uploaded.

        Capping the longest edge is preferred over a fixed ratio: a fixed
        ratio either leaves large images slow or shrinks small ones until
        their faces vanish.

        Returns the working image and the factor its coordinates must be
        divided by to return to source scale.
        """
        height, width = image.shape[:2]
        longest = max(height, width)

        if self._max_edge <= 0 or longest <= self._max_edge:
            return image, 1.0

        scale = self._max_edge / longest
        resized = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            # INTER_AREA is the right filter for shrinking; it avoids the
            # aliasing that would cost detections.
            interpolation=cv2.INTER_AREA,
        )
        return cast(RgbImage, resized), scale

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


def _to_box(location: tuple[int, int, int, int], scale: float) -> BoundingBox:
    """Convert a dlib location back to source-image coordinates."""
    top, right, bottom, left = location
    if scale == 1.0:
        return BoundingBox(top=top, right=right, bottom=bottom, left=left)

    return BoundingBox(
        top=round(top / scale),
        right=round(right / scale),
        bottom=round(bottom / scale),
        left=round(left / scale),
    )
