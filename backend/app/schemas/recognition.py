"""Schemas for recognition results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from backend.app.services.encoder import BoundingBox
from backend.app.services.recognizer import Recognition


class BoundingBoxModel(BaseModel):
    """Face position in source-image pixels.

    Uses dlib's ``(top, right, bottom, left)`` convention rather than
    ``(x, y, w, h)``, so the values pass through untranslated.
    """

    top: int
    right: int
    bottom: int
    left: int

    @classmethod
    def from_domain(cls, box: BoundingBox) -> BoundingBoxModel:
        return cls(top=box.top, right=box.right, bottom=box.bottom, left=box.left)


class FaceMatch(BaseModel):
    """One detected face and who it was matched to."""

    name: str = Field(description="Enrolled name, or 'Unknown'.")
    known: bool = Field(description="Whether the face matched an enrolled identity.")
    distance: float = Field(
        description="Euclidean distance to the nearest enrolment. Lower is closer."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Display-only score derived from distance. Not a calibrated "
            "probability; the tolerance threshold decides the match."
        ),
    )
    box: BoundingBoxModel

    @classmethod
    def from_domain(cls, recognition: Recognition) -> FaceMatch:
        return cls(
            name=recognition.name,
            known=recognition.is_known,
            # An unmatched face has infinite distance, which is not valid JSON.
            distance=(
                round(recognition.distance, 4) if recognition.distance != float("inf") else -1.0
            ),
            confidence=recognition.confidence,
            box=BoundingBoxModel.from_domain(recognition.box),
        )


class RecognitionResponse(BaseModel):
    """Result of recognising one frame.

    Returns coordinates rather than a rendered image, so the client draws the
    overlay and the server never re-encodes a JPEG.
    """

    faces: list[FaceMatch]
    count: int = Field(description="Number of faces detected.")
    processing_ms: float = Field(description="Server-side processing time.")

    @classmethod
    def from_domain(
        cls,
        recognitions: list[Recognition],
        *,
        processing_ms: float,
    ) -> RecognitionResponse:
        return cls(
            faces=[FaceMatch.from_domain(item) for item in recognitions],
            count=len(recognitions),
            processing_ms=round(processing_ms, 2),
        )
