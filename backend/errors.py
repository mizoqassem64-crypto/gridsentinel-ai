"""Stable, safe JSON error model for the GridSentinel inference API."""

from typing import Any, Dict, Optional

from . import API_VERSION


class ApiError(Exception):
    """An API error with an HTTP status, a stable code, and a safe message.

    ``message``/``detail`` are always static strings safe to return to API
    clients. Never interpolate exception text, file paths, hashes, model
    filenames, or environment values into them.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        detail: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail
        self.headers = headers or {}


def error_envelope(request_id: str, error: ApiError) -> Dict[str, Any]:
    """Stable JSON error envelope shared by every failure path."""
    return {
        "request_id": request_id,
        "api_version": API_VERSION,
        "status": "error",
        "error": {
            "code": error.code,
            "message": error.message,
            "detail": error.detail,
        },
    }


def invalid_json() -> ApiError:
    return ApiError(
        400,
        "invalid_json",
        "Request body is not well-formed JSON or contains non-finite "
        "JSON constants (NaN/Infinity are not permitted).",
    )


def bad_request(message: str) -> ApiError:
    return ApiError(400, "invalid_request", message)


def unauthorized() -> ApiError:
    return ApiError(
        401,
        "unauthorized",
        "Missing or invalid API key. Provide a key in the 'X-API-Key' "
        "request header. The key is never accepted in the JSON body.",
    )


def payload_too_large(max_bytes: int) -> ApiError:
    return ApiError(
        413,
        "payload_too_large",
        f"Request body exceeds the {max_bytes} byte limit.",
    )


def unsupported_media_type() -> ApiError:
    return ApiError(
        415,
        "unsupported_media_type",
        "Content-Type must be 'application/json'.",
    )


def validation_failed(detail: Any) -> ApiError:
    return ApiError(
        422,
        "validation_failed",
        "Request body failed strict telemetry validation.",
        detail,
    )


def rate_limited(retry_after: int) -> ApiError:
    return ApiError(
        429,
        "rate_limited",
        "Too many requests. Please retry after the Retry-After interval.",
        headers={"Retry-After": str(retry_after)},
    )


def not_found() -> ApiError:
    return ApiError(404, "not_found", "Resource not found.")


def method_not_allowed() -> ApiError:
    return ApiError(405, "method_not_allowed", "Method not allowed.")


def server_misconfigured() -> ApiError:
    return ApiError(
        503,
        "server_misconfigured",
        "The inference API is not configured (missing server-side API "
        "key). This is an operator error; the key is never read from "
        "client input.",
    )


def overloaded() -> ApiError:
    return ApiError(
        503,
        "server_overloaded",
        "Server is at request capacity. Retry shortly.",
    )


def internal_error() -> ApiError:
    return ApiError(
        500,
        "internal_error",
        "An unexpected internal error occurred. No details are "
        "disclosed for security; refer to the server-side log via the "
        "request id.",
    )