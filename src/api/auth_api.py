"""Authentication endpoints.  SRS FR i."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, g, jsonify, request
from flask_login import current_user, login_required

from src.auth import (
    ROLE_CAPABILITIES,
    ROLE_GRANTS,
    absolute_session_expiry,
    authenticate,
    audit_login,
    client_ip,
    password_problems,
    sign_in,
    sign_out,
)
from src.db import session_scope
from src.errors import ApiError
from src.models import ROLES, ROLE_LABELS, User

logger = logging.getLogger("sonicsentinel.api.auth")

bp = Blueprint("auth_api", __name__, url_prefix="/api/auth")


def _user_payload(user) -> dict:
    """What the client is told about itself. Never the hash, never the lockout internals."""
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name or user.username,
        "email": user.email,
        "role": user.role,
        "role_label": ROLE_LABELS.get(user.role, user.role),
        "must_change_password": bool(user.must_change_password),
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        "capabilities": sorted(ROLE_GRANTS.get(user.role, frozenset())),
    }


@bp.post("/login")
def login():
    """Exchange a username and password for a session.

    Failures are uniform: a wrong password, an unknown username, a locked account and a
    disabled account each return a stable ``code`` so the page can react specifically, but
    the response for an unknown username is byte-identical to a wrong password -- sign-in
    must not be usable to discover which usernames exist. The difference is recorded in the
    audit trail (FR lxxvi) instead, where an investigator can see it.
    """
    payload = request.get_json(silent=True) or request.form or {}
    username = str(payload.get("username", "")).strip()
    password = payload.get("password", "")
    remember = bool(payload.get("remember", False))

    if not username or not password:
        raise ApiError(
            "validation_error",
            "Enter both a username and a password.",
            details={"username": "required" if not username else "ok",
                     "password": "required" if not password else "ok"},
        )

    factory = current_app.config["SST_SESSION_FACTORY"]
    store = current_app.config["SST_CONFIG_STORE"]
    request_id = getattr(g, "request_id", None)

    # The per-IP and per-username login limits from config/auth.json (FR lxxviii).
    limiter_ip = current_app.extensions["sst_limiter_login_ip"]
    limiter_user = current_app.extensions["sst_limiter_login_user"]
    for limiter, key, label in (
        (limiter_ip, client_ip(), "this address"),
        (limiter_user, username.lower(), "this account"),
    ):
        result = limiter.check(key)
        if not result.allowed:
            with session_scope(factory) as s:
                audit_login(s, action="login_blocked", username=username, outcome="failure",
                            detail=f"rate limited ({label})", request_id=request_id)
            response = jsonify({"error": {
                "code": "rate_limited",
                "message": f"Too many sign-in attempts from {label}. "
                           f"Try again in {result.retry_after_seconds} seconds.",
                "details": {"retry_after": result.retry_after_seconds},
                "request_id": request_id,
            }})
            response.status_code = 429
            response.headers["Retry-After"] = str(result.retry_after_seconds)
            response.headers["X-Request-Id"] = request_id or "-"
            return response

    with session_scope(factory) as session:
        outcome = authenticate(session, username, password, store=store)
        if outcome.ok:
            row = outcome.user
            audit_login(session, action="login_success", username=row.username,
                        role=row.role, user_id=row.id, request_id=request_id,
                        detail=f"signed in from {client_ip()}")
            session.expunge(row)
        else:
            audit_login(session, action="login_failure", username=username, outcome="failure",
                        detail=outcome.code, request_id=request_id)

    if not outcome.ok:
        outcome.raise_if_failed()

    # A good sign-in clears that username's limiter, so a legitimate typo spree does not
    # leave the operator throttled after they get in.
    limiter_user.reset(username.lower())
    sign_in(outcome.user, remember=remember)

    return jsonify({
        "user": _user_payload(outcome.user),
        "session": {"expires_in_minutes": current_app.config["SST_SESSION_LIFETIME_MINUTES"]},
    })


@bp.post("/logout")
@login_required
def logout():
    """End the session and record it (FR lxxvi logs sign-outs too)."""
    user = current_user
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        audit_login(session, action="logout", username=user.username, role=user.role,
                    user_id=user.id, request_id=getattr(g, "request_id", None))
    sign_out()
    return jsonify({"ok": True})


@bp.get("/me")
@login_required
def me():
    """Who am I, and what may I do. The frontend uses ``capabilities`` to build its menus.

    The lists come from ``ROLE_CAPABILITIES`` -- the same source the server checks -- so a
    menu can never offer an action the server would refuse.
    """
    return jsonify({
        "user": _user_payload(current_user.row),
        "role": current_user.role,
        "capabilities": sorted(ROLE_GRANTS.get(current_user.role, frozenset())),
        "all_capabilities": sorted(ROLE_CAPABILITIES),
        "roles": [{"value": r, "label": ROLE_LABELS.get(r, r)} for r in ROLES],
        "session": {
            "expires_in_minutes": current_app.config["SST_SESSION_LIFETIME_MINUTES"],
            "expires_at": absolute_session_expiry(),
        },
    })


@bp.post("/password")
@login_required
def change_password():
    """Change your own password. FR i: credentials are managed, not handed out.

    Requires the current password even for an administrator, so a hijacked session cannot
    be turned into permanent account takeover.
    """
    from werkzeug.security import check_password_hash

    payload = request.get_json(silent=True) or request.form or {}
    current = payload.get("current_password", "")
    new = payload.get("new_password", "")
    store = current_app.config["SST_CONFIG_STORE"]

    if not check_password_hash(current_user.row.password_hash, current or ""):
        raise ApiError("invalid_credentials", "That is not your current password.",
                       status=401)
    if current == new:
        raise ApiError("validation_error", "The new password must differ from the current one.",
                       details={"new_password": "must differ"})
    problems = password_problems(new, store)
    if problems:
        raise ApiError("validation_error", "The new password does not meet the policy.",
                       details={"new_password": "; ".join(problems)})

    from src.auth import hash_password
    from src.db import record_audit

    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        row = session.get(User, current_user.id)
        row.password_hash = hash_password(new)
        row.must_change_password = False
        record_audit(session, action="password_change", actor=row, target_type="user",
                     target_id=row.username,
                     detail="changed their own password",
                     ip_address=client_ip(), request_id=getattr(g, "request_id", None))
    return jsonify({"ok": True})
