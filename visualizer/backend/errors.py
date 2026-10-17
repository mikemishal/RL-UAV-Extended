import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from .application_error import ApplicationError

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=ErrorDetail(code=code, message=message)).model_dump(),
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error(_request: Request, exc: ApplicationError) -> JSONResponse:
        logger.warning("%s: %s", exc.code, exc.message)
        return error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        errors = exc.errors()
        if errors:
            field = ".".join(str(part) for part in errors[0].get("loc", ()) if part != "body") or "request"
            message = f"{field}: {errors[0].get('msg', 'invalid value')}"
        else:
            message = "Request does not match the API contract."
        return error_response("INVALID_REQUEST", message, 422)

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return error_response("HTTP_ERROR", str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled API error", exc_info=exc)
        return error_response("INTERNAL_ERROR", "An unexpected backend error occurred. See server logs.", 500)
