"""Enrolment: register, list and remove known faces."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, Path, UploadFile, status

from backend.app.api.deps import EncoderDep, GalleryDep, SettingsDep
from backend.app.api.uploads import read_image_upload
from backend.app.schemas.common import ErrorResponse
from backend.app.schemas.face import IdentityListResponse, IdentityResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/faces", tags=["faces"])

# Names become primary keys and appear in URLs, so keep them to a predictable
# set rather than accepting arbitrary text.
NAME_PATTERN = r"^[\w][\w .'-]{0,63}$"

NameForm = Annotated[
    str,
    Form(
        min_length=1,
        max_length=64,
        pattern=NAME_PATTERN,
        description="Unique name to enrol the face under.",
    ),
]
NamePath = Annotated[str, Path(min_length=1, max_length=64, pattern=NAME_PATTERN)]


@router.post(
    "",
    response_model=IdentityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enrol a face",
    description=(
        "Registers a single face under a name. The image must contain exactly "
        "one face; zero or several is rejected rather than guessed at."
    ),
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ErrorResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse},
    },
)
async def enrol_face(
    settings: SettingsDep,
    encoder: EncoderDep,
    gallery: GalleryDep,
    name: NameForm,
    image: Annotated[UploadFile, File(description="Photo containing one face.")],
) -> IdentityResponse:
    picture = await read_image_upload(image, settings)

    # Detection and embedding are blocking C++ work; running them inline would
    # stall the event loop for every other request.
    face = await asyncio.to_thread(encoder.encode_single, picture)
    identity = await asyncio.to_thread(gallery.add, name, face.embedding)

    return IdentityResponse.from_domain(identity)


@router.get(
    "",
    response_model=IdentityListResponse,
    summary="List enrolled identities",
    description="Returns enrolled names and dates. Embeddings are never exposed.",
)
async def list_faces(gallery: GalleryDep) -> IdentityListResponse:
    identities = await asyncio.to_thread(gallery.list_all)
    return IdentityListResponse.from_domain(identities)


@router.delete(
    "/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an identity",
    description="Deletes an enrolled identity and its embedding.",
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def delete_face(gallery: GalleryDep, name: NamePath) -> None:
    await asyncio.to_thread(gallery.delete, name)
