"""Authentication and role-based access control (FR i, FR ii).

ROLE_CAPABILITIES is the permission matrix, checked on the server for every endpoint.
The user is reloaded on every request, so locking an account takes effect at once.
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


# Capability -> roles that have it.
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
    # Manual review (FR lvii-lxi). Maintenance is left out on purpose.
    "review_queue": frozenset({"audio_reviewer", "administrator"}),
    "review_decide": frozenset({"audio_reviewer", "administrator"}),
    # Alerts (FR liv-lvi)
    "view_alerts": frozenset({"security_operator", "administrator"}),
    "acknowledge_alerts": frozenset({"security_operator", "administrator"}),
    "alert_history": frozenset({"security_operator", "administrator"}),
    # Configuration and models (FR liii, lxxv, lxxx)
    "manage_models": frozenset({"maintenance_operator", "administrator"}),
    "edit_config": frozenset({"maintenance_operator", "administrator"}),
    # Administration.
    "manage_users": frozenset({"administrator"}),
    "read_audit": frozenset({"administrator"}),
    "retention_purge": frozenset({"administrator"}),
}

# Role -> capabilities, for menus and tests.
ROLE_GRANTS: Mapping[str, frozenset[str]] = {
    role: frozenset(cap for cap, roles in ROLE_CAPABILITIES.items() if role in roles)
    for role in ROLES
}


def has_capability(role: str | None, capability: str) -> bool:
    """True if the role has the capability."""
    if not role:
        return False
    return role in ROLE_CAPABILITIES.get(capability, frozenset())




class AuthUser:
    """Flask-Login wrapper around a User row, with ``can()`` for capability checks."""

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
    """The session factory for this app (tests can override it)."""
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
    manager.session_protection = "strong"

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
        # Send browsers to sign in, then back to the page they wanted.
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
    """Restrict an endpoint to users with a capability. Prefer this over roles_required."""
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
    """Allow users with the capability, or the owner of the row (FR lxvii).

    The handler checks ownership and returns 404, not 403.
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
    """List every way the password breaks the policy."""
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
    """Hash a password with PBKDF2-SHA256."""
    return generate_password_hash(password, method="pbkdf2:sha256")




class LoginOutcome:
    """Result of a login attempt: a user, or an error code and message."""

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
    """Check a username and password, with per-account lockout.

    Wrong password and unknown user get the same message so usernames can't be probed;
    the audit log records which it was.
    """
    from src.services.config import get_store

    store = store or get_store()
    max_attempts = int(store.auth_setting("login.max_failed_attempts", 5))
    lockout_minutes = float(store.auth_setting("login.lockout_minutes", 15))

    row = session.execute(select(User).where(User.username == (username or "").strip())
                          ).scalar_one_or_none()

    if row is None:
        # Take as long as a real check so timing doesn't reveal unknown usernames.
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
    """Log the verified user in."""
    login_user(AuthUser(user_row), remember=remember, fresh=True)


def sign_out() -> None:
    logout_user()


def absolute_session_expiry() -> str | None:
    """When this session expires, as ISO-8601."""
    from flask import session as flask_session

    if not flask_session:
        return None
    lifetime = current_app.config.get("SST_SESSION_LIFETIME_MINUTES", 480)
    return (utcnow() + dt.timedelta(minutes=float(lifetime))).isoformat()


def client_ip() -> str:
    """Client address for rate limits and the audit log.

    X-Forwarded-For is only used when SST_TRUSTED_PROXY_HOPS is set, and then we take the
    entry our proxy added (counting from the right). The leftmost entry is client-controlled.
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
    """Audit a sign-in or refusal. Takes plain values since a failed login may have no user."""
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
    # No user row for failed attempts, so fill in the actor columns by hand.
    if user_id is None:
        from src.models import AuditRecord

        row = session.execute(
            select(AuditRecord).order_by(AuditRecord.id.desc()).limit(1)
        ).scalar_one()
        row.actor_username = username
        row.actor_role = role


class _ActorStub:
    """Stand-in actor for record_audit when there is no User."""

    __slots__ = ("id", "username", "role")

    def __init__(self, user_id: int, username: str, role: str | None) -> None:
        self.id = user_id
        self.username = username
        self.role = role
