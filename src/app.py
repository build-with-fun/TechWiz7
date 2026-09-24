"""The application factory.  One place where the app is assembled.

Owner: sara.  SRS FR i (authentication), FR ii (roles), FR lxxvii (error handling),
FR lxxviii (monitoring), plus the non-functional requirements the web layer is on the hook
for: >= 99% uptime behaviour, no internals in an error message, and a boot that is honest
about what is and is not available.

Design rules this file holds to:

* **Everything configurable comes from a file.** Session lifetime, lockout, rate limits and
  the security headers are read from ``config/auth.json``; thresholds and class names come
  from ``config/``; alert rules from ``alert_rules/``. Nothing is a literal, so an
  evaluator editing a file mid-demo sees the change on the next request (SRS §1.8 rule 5).
* **The app boots with the models absent.** A missing model is a reported condition
  (``/api/health`` says so, ``model_unavailable`` is returned), never a crash at import or
  a hard failure to start. An evaluator restarting the box during a demo must get a
  sign-in page and a clear health badge, not a stack trace.
* **Predictors, the pipeline, the database and the clock are injected.** ``create_app``
  takes them, so the test suite runs the whole HTTP surface with no trained model, no
  microphone, and an in-memory database -- and the same code path is what serves the demo.
* **Every response carries ``X-Request-Id``**, success included, and every failure is the
  envelope from ``documentation/api_contract.md`` §1.1.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from flask import Flask, g, request, session

from src.db import (
    REPO_ROOT,
    StorageLayout,
    create_engine_for,
    create_schema,
    default_db_path,
    default_storage_dir,
    make_session_factory,
)
from src.errors import new_request_id, register_error_handlers

logger = logging.getLogger("sonicsentinel.app")

APP_NAME = "SonicSentinel AI"
APP_VERSION = "1.0.0"

#: The SRS §1.10 folder list, minus the code folders. Checked at startup and reported by
#: ``/api/health`` so a missing deliverable is visible in the demo rather than at hand-in.
REQUIRED_DATA_FOLDERS: tuple[str, ...] = (
    "config",
    "alert_rules",
    "database",
    "data",
    "documentation",
    "templates",
    "static",
    "tests",
    "audio_dataset",
    "sample_audio",
)

DEFAULT_CONFIG_DIR = "config"
DEFAULT_ALERT_RULES_DIR = "alert_rules"


def _configure_logging(app: Flask) -> None:
    """One log format, with the request id in it, so a line can be tied to a request.

    FR lxxviii wants failures findable. A log line that cannot be matched to the
    ``request_id`` the user was shown is not findable.
    """
    level = app.config.get("SST_LOG_LEVEL", "INFO")
    logging.getLogger("sonicsentinel").setLevel(level)
    if not logging.getLogger("sonicsentinel").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(message)s"
        ))
        logging.getLogger("sonicsentinel").addHandler(handler)


def _apply_security_headers(app: Flask, store) -> None:
    """The header set from ``config/auth.json``.

    Applied to every response, including errors -- a 403 page without a CSP is still a
    page. ``Permissions-Policy`` deliberately allows the microphone for this origin and
    nothing else, because the live monitoring page needs it and nothing else does.
    """
    settings = store.auth_config().get("security_headers", {})

    @app.after_request
    def _headers(response):
        for header, key in (
            ("Content-Security-Policy", "content_security_policy"),
            ("X-Content-Type-Options", "x_content_type_options"),
            ("X-Frame-Options", "x_frame_options"),
            ("Referrer-Policy", "referrer_policy"),
            ("Permissions-Policy", "permissions_policy"),
        ):
            value = settings.get(key)
            if value:
                response.headers[header] = value
        if request.is_secure:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        response.headers.setdefault("X-Request-Id", getattr(g, "request_id", "-"))
        return response


def _apply_request_context(app: Flask) -> None:
    """Give every request an id and start its audit breadcrumb."""

    @app.before_request
    def _start_request():
        # An inbound id is only trusted if it looks like one of ours: a client-supplied
        # string is echoed into logs, so it must not be able to inject anything.
        inbound = request.headers.get("X-Request-Id", "")
        g.request_id = inbound if (inbound and inbound.isalnum() and len(inbound) <= 32) \
            else new_request_id()
        g.request_started = time.perf_counter()
        g.current_user_row = None
        g.audit_actor = None

    @app.teardown_request
    def _end_request(exc):  # pragma: no cover - timing is diagnostic only
        if app.config.get("SST_LOG_SLOW_REQUESTS"):
            elapsed_ms = (time.perf_counter() - getattr(g, "request_started", time.perf_counter())) * 1000
            budgets = {"upload": 8000.0, "live": 3000.0}
            budget = budgets.get(app.config.get("SST_ACTIVE_JOB", "upload"), 3000.0)
            if elapsed_ms > budget:
                logger.warning(
                    "slow request: %s %s took %.0f ms (budget %.0f ms) [%s]",
                    request.method, request.path, elapsed_ms, budget, g.request_id,
                )


def _register_session_hooks(app: Flask) -> None:
    """Session cookie and lifetime, from ``config/auth.json`` rather than Flask defaults."""

    @app.before_request
    def _refresh_session_lifetime():
        if not session:
            return
        minutes = float(app.config["SST_SESSION_LIFETIME_MINUTES"])
        session.permanent = True
        session.modified = True
        app.permanent_session_lifetime = dt.timedelta(minutes=minutes)


def _register_blueprints(app: Flask) -> None:
    from src.api.auth_api import bp as auth_bp
    from src.api.health import bp as health_bp
    from src.api.pages import bp as pages_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(pages_bp)

    # Optional API slices. Each is registered only when its module is importable, so this
    # file does not have to change every time another engineer lands an endpoint group --
    # and a slice that is not there yet is reported by /api/health rather than crashing
    # the whole console.
    optional = (
        ("src.api.audio_api", "/api/audio"),
        ("src.api.events_api", "/api"),
        ("src.api.alerts_api", "/api"),
        ("src.api.reviews_api", "/api"),
        ("src.api.dashboard_api", "/api"),
        ("src.api.admin_api", "/api/admin"),
        ("src.api.live_api", "/api/live"),
        ("src.api.reports_api", "/api"),
    )
    registered: list[str] = []
    for module_path, url_prefix in optional:
        try:
            module = __import__(module_path, fromlist=["bp"])
        except ImportError as exc:
            logger.info("API slice %s not available yet (%s)", module_path, exc)
            continue
        bp = getattr(module, "bp", None)
        if bp is None:
            logger.warning("API slice %s has no 'bp' blueprint; skipped", module_path)
            continue
        app.register_blueprint(bp, url_prefix=url_prefix)
        registered.append(module_path)
    app.config["SST_REGISTERED_API_SLICES"] = registered


def _build_store(app: Flask):
    from src.services.config import ConfigStore, set_store

    store = ConfigStore(
        config_dir=app.config["SST_CONFIG_DIR"],
        alert_rules_dir=app.config["SST_ALERT_RULES_DIR"],
    )
    # Fail loudly at startup if the shipped configuration is broken: booting on a broken
    # threshold file and discovering it mid-demo is worse than refusing to start now.
    problems = store.validate()
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(f"configuration is invalid, refusing to start: {joined}")
    set_store(store)
    app.config["SST_CONFIG_STORE"] = store
    return store


def _load_models(app: Flask, pipeline: Any | None) -> Any | None:
    """Resolve the analysis pipeline, without ever failing the boot.

    Order: an explicitly injected pipeline (tests), then a real one loaded from disk, then
    nothing. The third case is a legitimate running state -- the server is up, the health
    endpoint reports why analysis is unavailable, and an analyse request gets a 503 with a
    sentence a user can read.
    """
    from src.services.pipeline import AnalysisPipeline, set_pipeline

    if pipeline is not None:
        set_pipeline(pipeline)
        app.config["SST_PIPELINE"] = pipeline
        app.config["SST_PIPELINE_ERROR"] = None
        return pipeline

    if app.config.get("SST_LOAD_MODELS", True) is False:
        set_pipeline(None)
        app.config["SST_PIPELINE"] = None
        app.config["SST_PIPELINE_ERROR"] = "model loading was disabled for this instance"
        return None

    try:
        loaded = AnalysisPipeline.load()
    except Exception as exc:
        # Recorded with its reason, reported by /api/health, and the app still starts.
        from src.services.pipeline import ModelsUnavailable

        reason = str(exc) if isinstance(exc, ModelsUnavailable) else f"{type(exc).__name__}: {exc}"
        logger.warning("analysis pipeline unavailable at start-up: %s", reason)
        set_pipeline(None)
        app.config["SST_PIPELINE"] = None
        app.config["SST_PIPELINE_ERROR"] = reason
        return None

    set_pipeline(loaded)
    app.config["SST_PIPELINE"] = loaded
    app.config["SST_PIPELINE_ERROR"] = None
    logger.info("analysis pipeline loaded: %s", loaded.models.describe())
    return loaded


def create_app(config: Mapping[str, Any] | None = None, **overrides: Any) -> Flask:
    """Build the application.

    ``config`` (and keyword ``overrides``) are applied on top of the defaults, so a test
    writes ``create_app(TESTING=True, SST_DB_PATH=":memory:")`` and gets a whole app with
    no model, no real database file and no microphone.
    """
    app = Flask(
        __name__,
        template_folder=str(REPO_ROOT / "templates"),
        static_folder=str(REPO_ROOT / "static"),
    )

    # Merge both ways of configuring the instance, so the bootstrap below sees an
    # overridden config directory no matter which argument carried it.
    supplied: dict[str, Any] = dict(config or {})
    supplied.update(overrides)

    defaults: dict[str, Any] = {
        "SST_REPO_ROOT": str(REPO_ROOT),
        "SST_CONFIG_DIR": str(REPO_ROOT / DEFAULT_CONFIG_DIR),
        "SST_ALERT_RULES_DIR": str(REPO_ROOT / DEFAULT_ALERT_RULES_DIR),
        "SST_DB_PATH": str(default_db_path()),
        "SST_STORAGE_DIR": str(default_storage_dir()),
        "SST_SESSION_FACTORY": None,
        "SST_LOG_LEVEL": os.environ.get("SST_LOG_LEVEL", "INFO"),
        "SST_LOG_SLOW_REQUESTS": True,
        "SST_LOAD_MODELS": True,
        "JSON_SORT_KEYS": False,
        "MAX_CONTENT_LENGTH": 64 * 1024 * 1024,
    }
    app.config.update(defaults)
    app.config.update(supplied)

    # config/auth.json decides the cookie settings that go into app.config, so the store
    # has to be readable before the app config is complete. Built once, against the final
    # directories, and reused for the rest of the boot.
    store = _build_store(app)

    app.secret_key = app.config.get("SECRET_KEY") or _secret_key(app.config)
    app.config["SST_SESSION_LIFETIME_MINUTES"] = store.auth_setting(
        "session.lifetime_minutes", 480
    )
    app.config["SST_REMEMBER_DAYS"] = store.auth_setting("session.remember_days", 14)
    app.config["SESSION_COOKIE_NAME"] = "sonicsentinel_session"
    app.config["SESSION_COOKIE_HTTPONLY"] = bool(
        store.auth_setting("session.cookie_httponly", True)
    )
    app.config["SESSION_COOKIE_SAMESITE"] = store.auth_setting(
        "session.cookie_samesite", "Lax"
    )
    app.config["SESSION_COOKIE_SECURE"] = bool(
        store.auth_setting("session.cookie_secure", False)
    )
    app.config["PERMANENT_SESSION_LIFETIME"] = dt.timedelta(
        minutes=float(app.config["SST_SESSION_LIFETIME_MINUTES"])
    )
    app.config["REMEMBER_COOKIE_DURATION"] = dt.timedelta(
        days=float(app.config["SST_REMEMBER_DAYS"])
    )

    _configure_logging(app)

    # -- storage and database ---------------------------------------------------------
    engine = app.config.get("SST_ENGINE")
    if engine is None:
        db_path = app.config["SST_DB_PATH"]
        engine = create_engine_for(db_path)
        app.config["SST_ENGINE"] = engine
    if app.config.get("SST_CREATE_SCHEMA", True):
        create_schema(engine)
    factory = app.config.get("SST_SESSION_FACTORY") or make_session_factory(engine)
    app.config["SST_SESSION_FACTORY"] = factory

    layout = StorageLayout(app.config["SST_STORAGE_DIR"])
    if app.config.get("SST_CREATE_STORAGE", True):
        layout.ensure()
    app.config["SST_STORAGE"] = layout

    # -- authentication -----------------------------------------------------------------
    from src.auth import _register_login_manager

    _register_login_manager(app)

    # -- models (never fatal) ------------------------------------------------------------
    _load_models(app, app.config.get("SST_PIPELINE_INJECT"))

    # -- HTTP behaviour -------------------------------------------------------------------
    _apply_request_context(app)
    _apply_security_headers(app, store)
    _register_session_hooks(app)
    register_error_handlers(app)
    _register_blueprints(app)

    @app.teardown_appcontext
    def _close_engine(exc):  # pragma: no cover - engine lives for the app's life
        return None

    app.config["SST_STARTED_AT"] = dt.datetime.now(dt.timezone.utc).isoformat()
    app.config["SST_INSTANCE_ID"] = uuid.uuid4().hex[:12]
    logger.info(
        "%s %s started: db=%s storage=%s pipeline=%s",
        APP_NAME, APP_VERSION, app.config["SST_DB_PATH"], app.config["SST_STORAGE_DIR"],
        "ready" if app.config.get("SST_PIPELINE") else "unavailable",
    )
    return app


def _secret_key(config: Mapping[str, Any]) -> str:
    """A stable secret key, or a loud temporary one.

    Stability matters: rotating it signs every user out. In development it is derived from
    the repository path so a restart does not drop the session; in a real deployment
    ``SST_SECRET_KEY`` (or ``SECRET_KEY``) must be set, and the health endpoint says so.
    """
    configured = os.environ.get("SST_SECRET_KEY") or os.environ.get("SECRET_KEY")
    if configured:
        return configured
    if config.get("TESTING"):
        return "testing-secret-key"
    import hashlib

    logger.warning(
        "SST_SECRET_KEY is not set, so a development key derived from the repository path is "
        "in use. Set SST_SECRET_KEY before deploying: sessions are invalidated whenever it "
        "changes, and a derived key is not a secret."
    )
    return hashlib.sha256(f"sonicsentinel-dev::{config.get('SST_REPO_ROOT')}".encode()).hexdigest()


def app_health() -> dict[str, Any]:  # pragma: no cover - used by CLI and health endpoint
    """A one-line summary for the CLI: is this instance able to work?"""
    from src.services.pipeline import pipeline_status

    status = pipeline_status()
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "pipeline_ready": bool(status.get("ready")),
        "reason": status.get("reason"),
    }
