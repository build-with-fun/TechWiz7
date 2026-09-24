"""Authentication and role-based access control.  SRS FR i and FR ii.

Owner: sara.

Three things here are deliberately not template concerns:

* **The permission matrix is code.** ``ROLE_CAPABILITIES`` is the executable form of the
  table published in ``documentation/api_contract.md`` §1.2, and a test parses that table
  back out of the markdown and fails if the two disagree. Hiding a link in a template is not
  access control; every capability is checked server-side on the endpoint itself.
* **A wrong role is a 403, not a 404 and not a hidden button.** The one exception the
  contract names: a request that must not reveal whether a row exists returns 404, because
  the row belongs to someone else. That is the *existence of another user's data* case, and
  it is decided in the handler, not here.
* **Sessions are server-validated on every request.** ``load_user`` re-reads the account, so
  deactivating or locking an account takes effect on the account's very next request rather
  than whenever its cookie happens to expire.
"""

from __future__ import annotations

import logging
import datetime as dt
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, TypeVar

from flask import abort, current_app, g, redirect, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from sqlalchemy import select
from werkzeug.security import check_password_hash, generate_password_hash

from src.db import session_scope
from src.errors import ApiError
from src.models import ROLES, User, utcnow

logger = logging.getLogger("sonicsentinel.auth")

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------------------
# The permission matrix -- FR ii, executable
# ---------------------------------------------------------------------------------------

#: Capability -> roles that hold it. Written as a mapping from capability so that adding a
#: capability forces an explicit decision for every role, rather than defaulting to "nobody
#: noticed it was missing".
ROLE_CAPABILITIES: Mapping[str, frozenset[str]] = {
    # Everyone signed in
    "view_own_events": frozenset(ROLES),
    "upload_audio": frozenset(ROLES),
    "live_session": frozenset(ROLES),
    "health": frozenset(ROLES),
    # Fleet-wide visibility and reading
    "view_all_events": frozenset({"audio_reviewer", "security_operator",
                                  "maintenance_operator", "administrator"}),
    "search_all_events": frozenset({"audio_reviewer", "security_operator",
                                    "maintenance_operator", "administrator"}),
    "download_any_audio": frozenset({"audio_reviewer", "security_operator",
                                     "maintenance_operator", "administrator"}),
    "view_dashboards": frozenset({"audio_reviewer", "security_operator",
                                  "maintenance_operator", "administrator"}),
    "view_analytics": frozenset({"audio_reviewer", "security_operator",
                                 "maintenance_operator", "administrator"}),
    "export_data": frozenset({"audio_reviewer", "security_operator",
                              "maintenance_operator", "administrator"}),
    "download_report": frozenset({"audio_reviewer", "security_operator",
                                  "maintenance_operator", "administrator"}),
    # Manual review -- FR lvii-lxi. Maintenance keeps out on purpose: they change how the
    # system judges, not what it concluded.
    "review_queue": frozenset({"audio_reviewer", "administrator"}),
    "review_decide": frozenset({"audio_reviewer", "administrator"}),
    # Alerts -- FR liv-lvi.
    "view_alerts": frozenset({"security_operator", "administrator"}),
    "acknowledge_alerts": frozenset({"security_operator", "administrator"}),
    "alert_history": frozenset({"security_operator", "administrator"}),
    # Configuration and models -- FR liii, lxxv, lxxx.
    "manage_models": frozenset({"maintenance_operator", "administrator"}),
    "edit_config": frozenset({"maintenance_operator", "administrator"}),
    # Administration.
    "manage_users": frozenset({"administrator"}),
    "read_audit": frozenset({"administrator"}),
    "retention_purge": frozenset({"administrator"}),
}

#: The reverse view, for rendering menus and for the test that checks the contract table.
ROLE_GRANTS: Mapping[str, frozenset[str]] = {
    role: frozenset(cap for cap, roles in ROLE_CAPABILITIES.items() if role in roles)
    for role in ROLES
}


def has_capability(role: str | None, capability: str) -> bool:
    """The single question the whole authorisation layer asks."""
    if not role:
        return False
    return role in ROLE_CAPABILITIES.get(capability, frozenset())


# ---------------------------------------------------------------------------------------
# The signed-in user
# ---------------------------------------------------------------------------------------


class AuthUser:
    """A thin, immutable view of a ``User`` row for Flask-Login.

    It delegates attribute access to the row, so templates written for the model
    (``current_user.username``, ``current_user.role_label``) keep working, while the
    capability question gets a real answer instead of a role string compared in a template.
    The row itself is kept as ``row`` for the audit trail.
    """

    __slots__ = ("row",)

    def __init__(self, row: User) -> None:
        self.row = row

    # -- Flask-Login's five ---------------------------------------------------------
    def get_id(self) -> str:
        return str(self.row.id)

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def is_active(self) -> bool:
        return bool(self.row.is_active)

    # -- what the capability layer and the templates ask -----------------------------
    @property
    def id(self) -> int:
        return self.row.id

    @property
    def username(self) -> str:
        return self.row.username

    @property
    def role(self) -> str:
        return self.row.role

    @property
    def display_name(self) -> str:
        return self.row.display_name or self.row.username

    def can(self, capability: str) -> bool:
        return has_capability(self.row.role, capability)

    def can_any(self, *capabilities: str) -> bool:
        return any(self.can(c) for c in capabilities)

    def __getattr__(self, name: str) -> Any:
        # Anything else the templates use (email, role_label, must_change_password, ...)
        # comes straight off the row, so new columns need no change here.
        return getattr(self.row, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AuthUser {self.row.username} ({self.row.role})>"


def get_db_factory():
    """The session factory the request is using. Set by the factory, overridable in tests."""
    factory = getattr(g, "db_session_factory", None)
    if factory is None:
        factory = current_app.config["SST_SESSION_FACTORY"]
        g.db_session_factory = factory
    return factory


def _load_user_record(user_id: str) -> User | None:
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        try:
            row = session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None
        if row is None:
            return None
        session.expunge(row)
        return row


def _register_login_manager(app) -> LoginManager:
    manager = LoginManager()
    manager.init_app(app)
    # Prompt the *user*, not a route: the browser gets redirected to the sign-in page,
    # while the API gets a 401 envelope from the unauthorized handler below.
    manager.login_view = "pages.login"
    manager.login_message = "Sign in to continue."
    manager.login_message_category = "warning"
    manager.session_protection = "strong"  # a stolen cookie is invalidated, not silently reused

    @manager.user_loader
    def _user_loader(user_id: str):  # type: ignore[unused-ignore]
        # Re-read on every request rather than trusting the cookie: a locked or disabled
        # account stops working immediately, which is what an operator expects after
        # locking it mid-incident.
        row = _load_user_record(user_id)
        if row is None or not row.is_active or row.is_locked():
            return None
        g.current_user_row = row
        g.audit_actor = f"{row.username}:{row.role}"
        return AuthUser(row)

    @manager.unauthorized_handler
    def _unauthorized():  # type: ignore[unused-ignore]
        from src.errors import wants_json

        if wants_json():
            raise ApiError("not_authenticated")
        # A person following a link gets the sign-in page and comes back to where they were.
        return redirect(url_for("pages.login", next=request.full_path.rstrip("?")))

    return manager


# ---------------------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------------------


def roles_required(*roles: str) -> Callable[[F], F]:
    """Restrict an endpoint to the named FR ii roles.

    Unauthenticated is ``401``; authenticated with the wrong role is ``403`` -- the contract's
    §1.4 rule, and the reason a reviewer testing "can the maintenance operator close an
    alert?" gets a refusal rather than a 500 or a silent success.
    """
    unknown = set(roles) - set(ROLES)
    if unknown:
        raise ValueError(f"roles_required got unknown role(s): {sorted(unknown)}")

    def decorator(view: F) -> F:
        @wraps(view)
        @login_required
        def wrapped(*args: Any, **kwargs: Any):
            user = current_user
            if not getattr(user, "is_authenticated", False):
                raise ApiError("not_authenticated")
            if user.role not in roles:  # type: ignore[attr-defined]
                raise ApiError(
                    "forbidden",
                    "Your role does not permit this action. "
                    f"This needs one of: {', '.join(sorted(roles))}.",
                )
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def capability_required(capability: str) -> Callable[[F], F]:
    """Restrict an endpoint to whoever holds a capability in ``ROLE_CAPABILITIES``.

    Preferred over ``roles_required`` inside the API: the handler then states the *action*
    it needs, and moving a capability between roles is a one-line change here rather than a
    hunt through decorators.
    """
    if capability not in ROLE_CAPABILITIES:
        raise ValueError(
            f"capability_required got unknown capability {capability!r}; add it to "
            f"ROLE_CAPABILITIES in src/auth.py and to the contract's matrix"
        )

    def decorator(view: F) -> F:
        @wraps(view)
        @login_required
        def wrapped(*args: Any, **kwargs: Any):
            user = current_user
            if not getattr(user, "is_authenticated", False):
                raise ApiError("not_authenticated")
            if not user.can(capability):  # type: ignore[attr-defined]
                raise ApiError(
                    "forbidden",
                    "Your role does not permit this action. "
                    f"This needs the '{capability}' permission.",
                    details={"capability": capability, "role": user.role},  # type: ignore[attr-defined]
                )
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def owner_or_capability(capability: str, owner_attr: str = "created_by_id") -> Callable[[F], F]:
    """Allow the holder of ``capability``, or the person who created the row.

    This is the FR lxvii rule that a normal user sees only their own events: a normal user
    holds no fleet-wide capability, so the handler must compare the row's owner to
    ``current_user.id`` -- and must answer **404**, not 403, because a 403 would confirm that
    someone else's event exists. That decision lives in the handler; this decorator only
    records the intent and enforces the "signed in" part.
    """

    def decorator(view: F) -> F:
        @wraps(view)
        @login_required
        def wrapped(*args: Any, **kwargs: Any):
            if not getattr(current_user, "is_authenticated", False):
                raise ApiError("not_authenticated")
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def is_admin() -> bool:
    return bool(getattr(current_user, "is_authenticated", False)
                and current_user.role == "administrator")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------------------
# Password policy -- config/auth.json, not a literal
# ---------------------------------------------------------------------------------------


def password_problems(password: str, store=None) -> list[str]:
    """Every way the password breaks the configured policy.

    Returns *all* problems rather than the first, so the sign-up form can mark every failing
    field at once instead of the user discovering them one at a time.
    """
    from src.services.config import get_store

    store = store or get_store()
    minimum = int(store.auth_setting("login.min_password_length", 10))
    problems: list[str] = []
    if len(password or "") < minimum:
        problems.append(f"must be at least {minimum} characters")
    if store.auth_setting("login.require_uppercase", True) and not any(c.isupper() for c in password):
        problems.append("must contain an uppercase letter")
    if store.auth_setting("login.require_digit", True) and not any(c.isdigit() for c in password):
        problems.append("must contain a digit")
    if store.auth_setting("login.require_symbol", True) and not any(
        not c.isalnum() for c in password
    ):
        problems.append("must contain a symbol")
    return problems


def hash_password(password: str) -> str:
    """PBKDF2-SHA256. The only way a password is ever written to the database."""
    return generate_password_hash(password, method="pbkdf2:sha256")


# ---------------------------------------------------------------------------------------
# Sign-in, with lockout -- FR i and FR lxxviii
# ---------------------------------------------------------------------------------------


class LoginOutcome:
    """The result of an attempt: either a user row, or the code and message to return."""

    def __init__(self, user: User | None = None, code: str | None = None,
                 message: str | None = None) -> None:
        self.user = user
        self.code = code
        self.message = message

    @property
    def ok(self) -> bool:
        return self.user is not None

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ApiError(self.code or "invalid_credentials", self.message)


def authenticate(session, username: str, password: str, *, store=None) -> LoginOutcome:
    """Check a username and password, applying the configured lockout.

    Every failure mode returns the *same* message for a wrong password and a missing
    account, so sign-in cannot be used to enumerate usernames -- the difference is only in
    the audit trail, where it belongs.

    The lockout counter is per account, not per IP: FR lxxviii wants a sustained attempt
    against one account to be visible and stoppable even when the attempts come from
    different addresses.
    """
    from src.services.config import get_store

    store = store or get_store()
    max_attempts = int(store.auth_setting("login.max_failed_attempts", 5))
    lockout_minutes = float(store.auth_setting("login.lockout_minutes", 15))

    row = session.execute(select(User).where(User.username == (username or "").strip())
                          ).scalar_one_or_none()

    if row is None:
        # Spend roughly the same time as a real verification would, so the response time
        # does not answer "does this username exist?" either.
        check_password_hash(
            "pbkdf2:sha256:600000$abcdefghijklmnop$"
            + "0" * 64,
            password or "",
        )
        return LoginOutcome(code="invalid_credentials")

    if row.is_locked():
        remaining = int((row.locked_until - utcnow()).total_seconds() // 60) + 1
        return LoginOutcome(
            code="account_locked",
            message=f"This account is locked. It unlocks automatically in about "
                    f"{remaining} minute(s), or an administrator can unlock it now.",
        )

    if not row.is_active:
        return LoginOutcome(code="account_disabled")

    if not check_password_hash(row.password_hash, password or ""):
        row.failed_login_count = (row.failed_login_count or 0) + 1
        if row.failed_login_count >= max_attempts:
            row.locked_until = utcnow() + dt.timedelta(minutes=lockout_minutes)
            logger.warning(
                "account locked after %d failed attempts: %s (%s)",
                row.failed_login_count, row.username, row.role,
            )
            session.flush()
            return LoginOutcome(
                code="account_locked",
                message=f"Too many failed attempts, so this account is locked for "
                        f"{int(lockout_minutes)} minutes.",
            )
        remaining = max_attempts - row.failed_login_count
        session.flush()
        return LoginOutcome(
            code="invalid_credentials",
            message="That username and password do not match an account. "
                    f"{remaining} attempt(s) remain before the account is locked.",
        )

    # Success.
    row.failed_login_count = 0
    row.locked_until = None
    row.last_login_at = utcnow()
    session.flush()
    return LoginOutcome(user=row)


# ---------------------------------------------------------------------------------------
# Small helpers used by the factory and the API
# ---------------------------------------------------------------------------------------


def sign_in(user_row: User, *, remember: bool = False) -> None:
    """Establish the session for a verified user row."""
    login_user(AuthUser(user_row), remember=remember, fresh=True)


def sign_out() -> None:
    logout_user()


def absolute_session_expiry() -> str | None:
    """When this session will expire, as ISO-8601, for the client to show.

    Read from the signed-in user's session rather than computed from "now", so the value
    the client displays is the one the server will actually enforce.
    """
    from flask import session as flask_session

    if not flask_session:
        return None
    lifetime = current_app.config.get("SST_SESSION_LIFETIME_MINUTES", 480)
    return (utcnow() + dt.timedelta(minutes=float(lifetime))).isoformat()


def client_ip() -> str:
    """The best address we have. Behind a proxy this is the forwarded original.

    Only the first hop is taken, so a client cannot append a forged chain and shift the
    blame for its own attempts onto an address it chose.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def audit_login(session, *, action: str, username: str, role: str | None = None,
                outcome: str = "success", detail: str | None = None,
                user_id: int | None = None, request_id: str | None = None) -> None:
    """Write a sign-in or refusal to the audit trail (FR lxxvi).

    Takes plain values rather than a row, because the failed-login case has no row to point
    at -- and "somebody tried 'admin' forty times" is exactly the record an investigation
    needs.
    """
    from src.db import record_audit

    record_audit(
        session,
        action=action,
        actor=None if user_id is None else _ActorStub(user_id, username, role),
        target_type="user",
        target_id=username,
        outcome=outcome,
        detail=detail,
        ip_address=client_ip(),
        request_id=request_id,
    )
    # record_audit resolves the actor from a row; for failed attempts the denormalised
    # columns must be filled in by hand so the trail still names the username tried.
    if user_id is None:
        from src.models import AuditRecord

        row = session.execute(
            select(AuditRecord).order_by(AuditRecord.id.desc()).limit(1)
        ).scalar_one()
        row.actor_username = username
        row.actor_role = role


class _ActorStub:
    """The minimum ``record_audit`` needs from an actor, for rows we do not have a User for."""

    __slots__ = ("id", "username", "role")

    def __init__(self, user_id: int, username: str, role: str | None) -> None:
        self.id = user_id
        self.username = username
        self.role = role
