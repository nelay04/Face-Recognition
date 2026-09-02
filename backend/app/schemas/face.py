"""Schemas for face enrolment and the identity gallery."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.services.gallery import Identity


class IdentityResponse(BaseModel):
    """One enrolled person.

    Deliberately omits the embedding: it is biometric data, and no client
    needs it to render a gallery.
    """

    name: str = Field(description="Unique name the face is enrolled under.")
    created_at: datetime = Field(description="When the identity was enrolled.")

    @classmethod
    def from_domain(cls, identity: Identity) -> IdentityResponse:
        return cls(name=identity.name, created_at=identity.created_at)


class IdentityListResponse(BaseModel):
    """Every enrolled identity."""

    identities: list[IdentityResponse]
    count: int = Field(description="Number of enrolled identities.")

    @classmethod
    def from_domain(cls, identities: list[Identity]) -> IdentityListResponse:
        return cls(
            identities=[IdentityResponse.from_domain(item) for item in identities],
            count=len(identities),
        )
