"""The error contract, in one place.

Owner: sara.  SRS FR lxxvii (error handling: invalid files, unsupported formats, decode
failure, model failure, database failure, report failure) and FR lxxviii (monitoring).

Two rules this module exists to enforce, both of them things a reviewer will try to break:

1. **Every failure looks the same.** All 4xx and 5xx responses from ``/api/`` are the
   envelope in ``documentation/api_contract.md`` §1.1 -- ``code``, ``message``,
   ``details``, ``request_id``. No stack trace, no SQL, no file path, no library name,
   ever. The exception goes to the log with the ``request_id``; the client gets the
   ``request_id`` so support can correlate without us leaking the internals.
2. **A failure is diagnosable without reading the client's screen.** Every 5xx and every
   permission refusal is logged with the request id, the route and the actor, and every
   permission refusal also lands in the audit trail (FR lxxvi ``access_denied``).

The ``code`` values are stable identifiers the frontend branches on; the list below is the
same one published in the contract, and ``tests/test_api_auth_rbac.py`` asserts the two
have not drifted apart.
"""

from __future__ import annotations

import logging
import secrets
import traceback
from typing import Any, Mapping

from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger("sonicsentinel.errors")

#: Every code the API is allowed to return. A code that is not here is a bug in a handler,
#: and the contract publishes this list, so adding one is a deliberate act.
ERROR_CODES: tuple[str, ...] = (
    # authentication and authorisation -- FR i, FR ii
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
    # audio ingestion -- FR iv-x, lxxiii, lxxiv
    "duplicate_audio",
    "quality_rejected",
    "quality_unusable",
    "decode_failed",
    # domain state -- FR lv, lxii
    "invalid_state_transition",
    "already_acknowledged",
    "alert_not_acknowledgeable",
    "already_reviewed",
    # configuration and models -- FR liii, lxxv, lxxx
    "config_invalid",
    "model_unavailable",
    # exports and reports -- FR lxix, lxx
    "export_too_large",
    "report_failed",
    # storage
    "storage_error",
    "database_error",
    # catch-all
    "internal_error",
)

#: The status code each error code is returned with. Kept beside the code list so the two
#: cannot disagree, and so a handler says ``raise ApiError("duplicate_audio")`` and gets the
#: right status without restating it.
DEFAULT_STATUS: Mapping[str, int] = {
    "invalid_credentials": 401,
    "account_locked": 423,
    "account_disabled": 403,
    "not_authenticated": 401,
    "forbidden": 403,
    "not_found": 404,
    "method_not_allowed": 405,
    "conflict": 409,
    "validation_error": 400,
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
    "export_too_large": 413,
    "report_failed": 500,
    "storage_error": 500,
    "database_error": 500,
    "internal_error": 500,
}

#: Safe, plain sentences for codes thrown without a message. Each one is written to be
#: readable aloud to a user (faris audits this) and to describe no internals.
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
    "export_too_large": "That export is too large to prepare at once. Narrow the filters.",
    "report_failed": "The report could not be produced. The event itself is unaffected.",
    "storage_error": "The audio storage is not available right now.",
    "database_error": "The service is temporarily unable to read its records.",
    "internal_error": "Something went wrong on our side. The error has been logged.",
}

#: Which HTTP status each code maps to when the client did not choose. Used to translate a
#: bare Werkzeug ``HTTPException`` into our envelope without leaking its description text
#: (Werkzeug's messages name routes and methods; ours must not).
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


#: The statuses a browser gets an HTML page for. Beyond these the client gets the JSON
#: envelope even on a page route, because there is no page worth writing for them and a
#: half-designed one is worse than a clear answer. The frontend ships one template per entry.
_RENDERABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 413, 415,
                                                  422, 423, 429, 500, 503})


class ApiError(Exception):
    """A failure a handler chose deliberately, with a code the frontend can branch on.

    Raising one of these is how a handler says "this is expected, tell the user this" as
    opposed to letting something explode and become an opaque 500. The message is written
    by us and is safe to show; ``details`` carries field-level validation help and must
    never contain a path, a query or an exception's text.
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
        #: Extra response headers this failure needs -- a ``429`` is not much use without its
        #: ``Retry-After``, and a ``405`` should say which methods it does allow. Carried on
        #: the exception so the handler that raises it does not have to build a Response by
        #: hand and thereby risk bypassing the shared envelope.
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


# ---------------------------------------------------------------------------------------
# Named constructors -- so handlers read like the contract, not like plumbing
# ---------------------------------------------------------------------------------------


def validation_error(message: str | None = None, **fields: Any) -> ApiError:
    """422-body shaped problem. ``validator_error("Too short", confidence="must be 0-1")``."""
    return ApiError("validation_error", message, details=fields or None)


def not_found(what: str = "item") -> ApiError:
    """404. Also used deliberately where naming the resource would leak another user's data
    -- see the contract §1.4, rule: *capability* is 403, *existence of another's data* is 404.
    """
    return ApiError("not_found", f"That {what} does not exist.")


def forbidden(what: str | None = None) -> ApiError:
    """403, with the role named so the message is actionable rather than a wall."""
    message = "Your role does not permit this action."
    if what:
        message = f"Your role does not permit this action: {what}."
    return ApiError("forbidden", message)


# ---------------------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------------------


def new_request_id() -> str:
    """A short opaque id, safe to show a user and to quote in a support message."""
    return secrets.token_hex(8)


def current_request_id() -> str:
    return getattr(g, "request_id", None) or "-"


def wants_json() -> bool:
    """JSON for the API and for any client that asked for it; HTML for a browser page.

    A person who opens ``/dashboard`` and hits an error should get a page, not a JSON blob;
    ``fetch()`` from the same app always sends ``Accept: application/json``. This is what
    makes one error layer serve both, so the two can never disagree about a status code.
    """
    if request.path.startswith("/api/"):
        return True
    accept = request.accept_mimetypes
    return accept["application/json"] > accept["text/html"]


# ---------------------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------------------


def _log_failure(err: ApiError, *, exc_info: bool = False) -> None:
    """Log at a level that matches what it is. A 401 is not an incident; a 500 is."""
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
        # exc_info gives the traceback in the log -- which is where it belongs, and the one
        # place it must never reach the client from.
        logger.error("request failed: %s", context, exc_info=exc_info)
    elif err.status == 403 or err.status == 401 or err.status == 429:
        logger.warning("request refused: %s", context)
    else:
        logger.info("request rejected: %s", context)


def _html_or_json(err: ApiError, request_id: str) -> Response:
    """Render the error the way the client asked for it.

    A browser gets the page for its status (``errors/404.html``, ``errors/500.html``, ...).
    The API always gets the JSON envelope, and so does any status we have no page for -- a
    missing template must not turn a clean 403 into a blank screen or a 500.
    """
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
            # No template yet, or a template that itself failed. Falling back to JSON keeps
            # the error contract intact instead of turning a 404 into a 500.
            response = jsonify(err.to_dict(request_id))
        else:
            response = Response(html, mimetype="text/html")
    response.status_code = err.status
    response.headers["X-Request-Id"] = request_id
    for header, value in err.headers.items():
        response.headers[header] = value
    return response


def _audit_denial(err: ApiError, request_id: str) -> None:
    """FR lxxvi: a refusal is an event worth recording.

    Written with its own short transaction so it cannot be rolled back by whatever the
    request was doing when it failed, and it never raises: an audit write that breaks the
    error response would turn a 403 into a 500.
    """
    if err.status not in (401, 403, 429):
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
                action="access_denied",
                actor=getattr(g, "current_user_row", None),
                target_type="endpoint",
                target_id=request.path,
                outcome="failure",
                detail=f"{err.code} on {request.method} {request.path}",
                ip_address=request.headers.get("X-Forwarded-For", request.remote_addr),
                request_id=request_id,
            )
    except Exception:  # pragma: no cover - auditing must never mask the real failure
        logger.exception("could not record an access-denied audit row")


def register_error_handlers(app: Flask) -> None:
    """Attach the envelope to the app. Called once, by the factory."""

    @app.errorhandler(ApiError)
    def _handle_api_error(err: ApiError):  # type: ignore[unused-ignore]
        request_id = current_request_id()
        _log_failure(err)
        _audit_denial(err, request_id)
        return _html_or_json(err, request_id)

    @app.errorhandler(HTTPException)
    def _handle_http_exception(err: HTTPException):  # type: ignore[unused-ignore]
        # Werkzeug's own errors (404 on an unknown route, 405, 413 from the body parser).
        # Its description can name a route or a limit, so only our text is sent.
        code = _STATUS_TO_CODE.get(err.code or 500, "internal_error")
        wrapped = ApiError(code, status=err.code or 500)
        _log_failure(wrapped)
        return _html_or_json(wrapped, current_request_id())

    @app.errorhandler(Exception)
    def _handle_unexpected(err: Exception):  # type: ignore[unused-ignore]
        # The last line of defence. The traceback goes to the log with the request id; the
        # client gets a sentence and the id. FR lxxvii also wants "database failure" and
        # "report failure" distinguished, so those get their own codes rather than being
        # flattened into internal_error.
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
    """The published list, for a test to compare against the contract."""
    return ERROR_CODES
