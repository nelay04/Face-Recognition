"""Application error types and their HTTP handlers."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from backend.app.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that map onto a known HTTP response.

    Raising these keeps route handlers free of HTTP plumbing while still
    producing a consistent error body.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message


class ServiceUnavailableError(AppError):
    """A dependency the service needs is not ready."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "The service is not ready to accept traffic."


class InvalidImageError(AppError):
    """The upload could not be decoded as a usable image."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_image"
    message = "The uploaded file is not a readable image."


class PayloadTooLargeError(AppError):
    """The upload exceeds the configured size limit."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"
    message = "The uploaded file is too large."


class NoFaceDetectedError(AppError):
    """Enrolment needs exactly one face and found none."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "no_face_detected"
    message = "No face was found in the image."


class MultipleFacesError(AppError):
    """Enrolment needs exactly one face and found several."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "multiple_faces"
    message = "The image contains more than one face; enrol a single face."


class IdentityExistsError(AppError):
    """An identity is already enrolled under that name."""

    status_code = status.HTTP_409_CONFLICT
    code = "identity_exists"
    message = "An identity with that name is already enrolled."


class IdentityNotFoundError(AppError):
    """No identity is enrolled under that name."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "identity_not_found"
    message = "No identity with that name is enrolled."


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that render errors as :class:`ErrorResponse`."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=exc.code, message=exc.message).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Log the traceback but never leak internals to the client.
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                code="internal_error",
                message="An unexpected error occurred.",
            ).model_dump(),
        )
