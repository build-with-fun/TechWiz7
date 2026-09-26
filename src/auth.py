"""Authentication and role-based access control (FR i, FR ii).

ROLE_CAPABILITIES is the permission matrix, checked on the server for every endpoint;
hiding a link in a template is not access control. A wrong role gets 403. Handlers that
must not reveal whether another user's row exists answer 404 instead. load_user re-reads
the account on every request, so locking or deactivating it takes effect immediately.
"""

from __future__ import annotations

import logging
import datetime as dt
from functools import wraps
from typing import Any, Callable, Mapping, TypeVar

from flask import current_app, g, redirect, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from sqlalchemy import select
from werkzeug.security import check_password_hash, generate_password_hash

from src.db import session_scope
from src.errors import ApiError
from src.models import ROLES, User, utcnow

logger = logging.getLogger("sonicsentinel.auth")

F = TypeVar("F", bound=Callable[..., Any])


#: Capability -> roles. Keyed by capability so a new one needs an explicit decision for
#: every role.
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
    "export_data": frozenset({"administrator"}),
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




class AuthUser:
    """Flask-Login view of a User row: templates keep reading row attributes, and ``can()`` answers
    capability questions.
    """

    __slots__ = ("row",)

    def __init__(self, row: User) -> None:
        self.row = row

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
        # Other attributes (email, role_label, ...) come straight off the row.
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
    # Browsers are redirected to sign in; API calls get a 401 from the handler below.
    manager.login_view = "auth.login"
    manager.login_message = "Sign in to continue."
    manager.login_message_category = "warning"
    manager.session_protection = "strong"  # a stolen cookie is invalidated, not silently reused

    @manager.user_loader
    def _user_loader(user_id: str):  # type: ignore[unused-ignore]
        # Re-read on every request so a locked account stops working immediately.
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
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))

    return manager




def roles_required(*roles: str) -> Callable[[F], F]:
    """Restrict an endpoint to named roles: 401 when signed out, 403 for the wrong role."""
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
    """Restrict an endpoint to holders of a capability. Preferred over roles_required, so moving a
    capability is a one-line change.
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
    """Allow a capability holder or the row's creator (FR lxvii: normal users see only their own events).

    The handler compares the owner and answers 404, not 403, so a refusal does not confirm that
    someone else's event exists.
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




def password_problems(password: str, store=None) -> list[str]:
    """Every way the password breaks the configured policy, so the form can flag them all at once.
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
    """Check a username and password with the configured per-account lockout.

    A wrong password and an unknown account get the same message, so sign-in cannot be used to
    list usernames; the audit trail records the difference. The lockout is per account so an
    attack spread across addresses is still stopped (FR lxxviii).
    """
    from src.services.config import get_store

    store = store or get_store()
    max_attempts = int(store.auth_setting("login.max_failed_attempts", 5))
    lockout_minutes = float(store.auth_setting("login.lockout_minutes", 15))

    row = session.execute(select(User).where(User.username == (username or "").strip())
                          ).scalar_one_or_none()

    if row is None:
        # Spend about as long as a real verification, so timing does not reveal whether the
        # username exists.
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

    row.failed_login_count = 0
    row.locked_until = None
    row.last_login_at = utcnow()
    session.flush()
    return LoginOutcome(user=row)




def sign_in(user_row: User, *, remember: bool = False) -> None:
    """Establish the session for a verified user row."""
    login_user(AuthUser(user_row), remember=remember, fresh=True)


def sign_out() -> None:
    logout_user()


def absolute_session_expiry() -> str | None:
    """When this session expires (ISO-8601), read from the session the server will enforce."""
    from flask import session as flask_session

    if not flask_session:
        return None
    lifetime = current_app.config.get("SST_SESSION_LIFETIME_MINUTES", 480)
    return (utcnow() + dt.timedelta(minutes=float(lifetime))).isoformat()


def client_ip() -> str:
    """The caller's address, used for per-IP rate limits and the audit trail.

    X-Forwarded-For is only trusted when SST_TRUSTED_PROXY_HOPS says proxies are in front,
    and then the entry our own proxy appended is used (counting from the right). The first
    entry is whatever the client chose to send: trusting it, as this function did until
    26 Sep, let a client dodge the per-IP login limit with a new fake address per attempt.
    """
    hops = int(current_app.config.get("SST_TRUSTED_PROXY_HOPS", 0) or 0)
    forwarded = [part.strip() for part in request.headers.get("X-Forwarded-For", "").split(",")
                 if part.strip()]
    if hops > 0 and len(forwarded) >= hops:
        return forwarded[-hops]
    return request.remote_addr or "unknown"


def audit_login(session, *, action: str, username: str, role: str | None = None,
                outcome: str = "success", detail: str | None = None,
                user_id: int | None = None, request_id: str | None = None) -> None:
    """Write a sign-in or refusal to the audit trail; takes plain values because a failed login has
    no user row.
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
    # Failed attempts have no row, so the denormalised actor columns are filled in by hand.
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
