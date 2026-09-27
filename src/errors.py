"""API error responses (FR lxxvii) and auditing of refusals (FR lxxvi).

Every 4xx/5xx from /api/ returns {code, message, details, request_id}. Tracebacks only go
to the log. The frontend relies on the ``code`` values staying the same.
"""

from __future__ import annotations

import logging
import secrets
import traceback
from typing import Any, Mapping

from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger("sonicsentinel.errors")

# All codes the API can return.
ERROR_CODES: tuple[str, ...] = (
    # authentication and authorisation (FR i, FR ii)
    "invalid_credentials",
    "account_locked",
    "account_disabled",
    "not_authenticated",
    "forbidden",
    # resources
    "not_found",
    "method_not_allowed",
    "conflict",
    # request shape
    "validation_error",
    "unsupported_media_type",
    "file_too_large",
    "rate_limited",
    # audio ingestion (FR iv-x, lxxiii, lxxiv)
    "duplicate_audio",
    "quality_rejected",
    "quality_unusable",
    "decode_failed",
    # domain state (FR lv, lxii)
    "invalid_state_transition",
    "already_acknowledged",
    "alert_not_acknowledgeable",
    "already_reviewed",
    # configuration and models (FR liii, lxxv, lxxx)
    "config_invalid",
    "model_unavailable",
    "pipeline_unavailable",
    # exports and reports (FR lxix, lxx)
    "export_too_large",
    "report_failed",
    # storage
    "no_audio",
    "storage_error",
    "database_error",
    # catch-all
    "internal_error",
)

# HTTP status for each code.
DEFAULT_STATUS: Mapping[str, int] = {
    "invalid_credentials": 401,
    "account_locked": 423,
    "account_disabled": 403,
    "not_authenticated": 401,
    "forbidden": 403,
    "not_found": 404,
    "method_not_allowed": 405,
    "conflict": 409,
    # A well-formed request with an invalid value is 422, not 400.
    "validation_error": 422,
    "unsupported_media_type": 415,
    "file_too_large": 413,
    "rate_limited": 429,
    "duplicate_audio": 409,
    "quality_rejected": 422,
    "quality_unusable": 422,
    "decode_failed": 422,
    "invalid_state_transition": 409,
    "already_acknowledged": 409,
    "alert_not_acknowledgeable": 409,
    "already_reviewed": 409,
    "config_invalid": 422,
    "model_unavailable": 503,
    "pipeline_unavailable": 503,
    "export_too_large": 413,
    "report_failed": 500,
    "no_audio": 410,
    "storage_error": 500,
    "database_error": 500,
    "internal_error": 500,
}

# Default messages for codes raised without one.
DEFAULT_MESSAGES: Mapping[str, str] = {
    "invalid_credentials": "That username and password do not match an account.",
    "account_locked": "This account is locked after too many failed sign-in attempts. "
                      "It unlocks automatically, or an administrator can unlock it.",
    "account_disabled": "This account has been disabled. Ask an administrator to enable it.",
    "not_authenticated": "Sign in to continue.",
    "forbidden": "Your role does not permit this action.",
    "not_found": "That item does not exist.",
    "method_not_allowed": "That action is not available for this item.",
    "conflict": "That change conflicts with the item's current state.",
    "validation_error": "Some of the details sent were not valid.",
    "unsupported_media_type": "That file type is not supported. "
                              "Upload a WAV, MP3, FLAC, OGG, M4A or AAC file.",
    "file_too_large": "That file is larger than the configured upload limit.",
    "rate_limited": "Too many requests. Wait a moment and try again.",
    "duplicate_audio": "That recording has already been submitted.",
    "quality_rejected": "That recording's audio quality is too poor to analyse reliably.",
    "quality_unusable": "That recording cannot be analysed: the audio quality is unusable.",
    "decode_failed": "That recording could not be read. It may be damaged or truncated.",
    "invalid_state_transition": "That item is not in a state where this action is allowed.",
    "already_acknowledged": "That alert has already been acknowledged.",
    "alert_not_acknowledgeable": "That alert cannot be acknowledged in its current state.",
    "already_reviewed": "That item has already been reviewed.",
    "config_invalid": "That configuration is not valid, so it was not applied. "
                      "The running settings are unchanged.",
    "model_unavailable": "The detection models are not available right now. "
                         "Try again shortly.",
    "pipeline_unavailable": "The analysis pipeline is not available right now. "
                            "Try again shortly.",
    "export_too_large": "That export is too large to prepare at once. Narrow the filters.",
    "report_failed": "The report could not be produced. The event itself is unaffected.",
    "no_audio": "This recording is no longer stored. Its analysis history is still available.",
    "storage_error": "The audio storage is not available right now.",
    "database_error": "The service is temporarily unable to read its records.",
    "internal_error": "Something went wrong on our side. The error has been logged.",
}

# Status -> code for plain Werkzeug exceptions (their messages are not shown).
_STATUS_TO_CODE: Mapping[int, str] = {
    400: "validation_error",
    401: "not_authenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    406: "validation_error",
    408: "internal_error",
    409: "conflict",
    413: "file_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    423: "account_locked",
    429: "rate_limited",
    500: "internal_error",
    501: "internal_error",
    503: "model_unavailable",
}


# Statuses that have an HTML error page.
_RENDERABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 413, 415,
                                                  422, 423, 429, 500, 503})


class ApiError(Exception):
    """An expected API error with a code and a message that is safe to show.

    ``details`` is for field-level validation help; never put paths or exception text in it.
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        status: int | None = None,
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if code not in DEFAULT_STATUS:
            raise ValueError(
                f"unknown error code {code!r}; add it to ERROR_CODES, DEFAULT_STATUS and "
                f"DEFAULT_MESSAGES in src/errors.py (and to the contract's code list)"
            )
        self.code = code
        self.message = message or DEFAULT_MESSAGES.get(code) or DEFAULT_MESSAGES["internal_error"]
        self.status = status or DEFAULT_STATUS[code]
        self.details: dict[str, Any] = dict(details or {})
        # Extra headers the response needs (Retry-After for 429, Allow for 405).
        self.headers: dict[str, str] = dict(headers or {})
        super().__init__(f"{self.status} {self.code}: {self.message}")

    def to_dict(self, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "request_id": request_id,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ApiError {self.status} {self.code}>"




def validation_error(message: str | None = None, **fields: Any) -> ApiError:
    """422 listing the bad fields, e.g. validation_error("Too short", confidence="must be 0-1")."""
    return ApiError("validation_error", message, details=fields or None)


def not_found(what: str = "item") -> ApiError:
    """404. Also used when a 403 would reveal that someone else's data exists."""
    return ApiError("not_found", f"That {what} does not exist.")


def forbidden(what: str | None = None) -> ApiError:
    """403 that names the user's role."""
    message = "Your role does not permit this action."
    if what:
        message = f"Your role does not permit this action: {what}."
    return ApiError("forbidden", message)




def new_request_id() -> str:
    """Short random request id."""
    return secrets.token_hex(8)


def current_request_id() -> str:
    return getattr(g, "request_id", None) or "-"


def wants_json() -> bool:
    """True for API calls and fetch(), False for normal page loads."""
    if request.path.startswith("/api/"):
        return True
    accept = request.accept_mimetypes
    return accept["application/json"] > accept["text/html"]




def _log_failure(err: ApiError, *, exc_info: bool = False) -> None:
    """Log 5xx as errors and 4xx at a lower level."""
    context = {
        "request_id": current_request_id(),
        "method": request.method,
        "path": request.path,
        "code": err.code,
        "status": err.status,
    }
    actor = getattr(g, "audit_actor", None)
    if actor:
        context["actor"] = actor
    if err.status >= 500:
        # Traceback to the log only.
        logger.error("request failed: %s", context, exc_info=exc_info)
    elif err.status == 403 or err.status == 401 or err.status == 429:
        logger.warning("request refused: %s", context)
    else:
        logger.info("request rejected: %s", context)


def _html_or_json(err: ApiError, request_id: str) -> Response:
    """Return HTML or JSON depending on the client. Falls back to JSON if the template fails."""
    if wants_json() or not err.status in _RENDERABLE_STATUSES:
        response = jsonify(err.to_dict(request_id))
    else:
        from flask import render_template

        try:
            html = render_template(
                f"errors/{err.status}.html",
                status=err.status,
                code=err.code,
                message=err.message,
                details=err.details,
                request_id=request_id,
            )
        except Exception:
            # Missing or broken template: use JSON rather than turning a 404 into a 500.
            response = jsonify(err.to_dict(request_id))
        else:
            response = Response(html, mimetype="text/html")
    response.status_code = err.status
    response.headers["X-Request-Id"] = request_id
    for header, value in err.headers.items():
        response.headers[header] = value
    return response


# Upload/live codes meaning the analysis couldn't run. Audited for the anomaly checks
# (FR lxxviii).
_MODEL_FAILURE_CODES = {"model_unavailable", "pipeline_unavailable"}


def _audit_action_for(err: ApiError) -> str | None:
    if err.status in (401, 403, 429):
        return "access_denied"
    is_analysis = request.path == "/api/audio/upload" or (
        request.path.startswith("/api/live/sessions/") and request.path.endswith("/windows"))
    if not is_analysis:
        return None
    if err.code in _MODEL_FAILURE_CODES:
        return "model_failure"
    if err.code == "duplicate_audio":
        return "audio_duplicate_rejected"
    return "audio_upload"


def _audit_denial(err: ApiError, request_id: str) -> None:
    """Audit refusals and failed analyses (FR lxxvi, lxxviii). Never raises."""
    try:
        action = _audit_action_for(err)
    except RuntimeError:  # outside a request context
        return
    if action is None:
        return
    from src.auth import get_db_factory

    try:
        db = get_db_factory()
    except Exception:  # no app context, or an app with no database configured
        return
    if db is None:
        return
    try:
        from src.db import record_audit, session_scope

        with session_scope(db) as session:
            record_audit(
                session,
                action=action,
                actor=getattr(g, "current_user_row", None),
                target_type="endpoint",
                target_id=request.path,
                outcome="failure",
                detail=f"{err.code} on {request.method} {request.path}",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
                request_id=request_id,
            )
    except Exception:  # pragma: no cover - don't hide the original error
        logger.exception("could not record an access-denied audit row")


def register_error_handlers(app: Flask) -> None:
    """Register the error handlers."""

    @app.errorhandler(ApiError)
    def _handle_api_error(err: ApiError):  # type: ignore[unused-ignore]
        request_id = current_request_id()
        _log_failure(err)
        _audit_denial(err, request_id)
        return _html_or_json(err, request_id)

    @app.errorhandler(HTTPException)
    def _handle_http_exception(err: HTTPException):  # type: ignore[unused-ignore]
        # Werkzeug errors (unknown route, 405, 413), with our own message.
        code = _STATUS_TO_CODE.get(err.code or 500, "internal_error")
        wrapped = ApiError(code, status=err.code or 500)
        _log_failure(wrapped)
        return _html_or_json(wrapped, current_request_id())

    @app.errorhandler(Exception)
    def _handle_unexpected(err: Exception):  # type: ignore[unused-ignore]
        # Anything unexpected. Database and storage errors get their own codes (FR lxxvii).
        code = "internal_error"
        status = 500
        name = type(err).__module__ + "." + type(err).__name__
        if "sqlalchemy" in name.lower() or "sqlite3" in name.lower():
            code = "database_error"
        elif isinstance(err, (OSError, IOError)) and request.path.startswith("/api/"):
            code = "storage_error"

        wrapped = ApiError(code, status=status)
        logger.error(
            "unhandled exception on %s %s [%s]: %s\n%s",
            request.method,
            request.path,
            current_request_id(),
            name,
            traceback.format_exc(),
        )
        response = jsonify(wrapped.to_dict(current_request_id()))
        response.status_code = status
        response.headers["X-Request-Id"] = current_request_id()
        return response


def error_codes() -> tuple[str, ...]:
    """The published error codes."""
    return ERROR_CODES
