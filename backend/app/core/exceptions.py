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
