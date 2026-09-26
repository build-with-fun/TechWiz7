"""The application factory.

Settings come from files (config/auth.json for sessions, lockout, rate limits and headers;
config/ and alert_rules/ for thresholds and rules), so an edit takes effect without code
changes (SRS 1.8 rule 5). The app starts even when a model is missing: /api/health says
why, and analysis requests get a 503 with a readable message. The pipeline, database and
predictors can be injected, which is how the tests run the whole HTTP surface. Every
response carries X-Request-Id.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import hmac
import secrets
import time
import uuid
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

#: The SRS 1.10 folder list, checked at startup and reported by /api/health.
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
    """One log format including the request id, so a failure can be matched to what the user saw."""
    level = app.config.get("SST_LOG_LEVEL", "INFO")
    logging.getLogger("sonicsentinel").setLevel(level)
    if not logging.getLogger("sonicsentinel").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s %(message)s"
        ))
        logging.getLogger("sonicsentinel").addHandler(handler)


def _apply_security_headers(app: Flask, store) -> None:
    """Security headers from config/auth.json, applied to every response including errors.

    Permissions-Policy allows the microphone for this origin only; the live page needs it.
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
        # Only an id that looks like ours is reused: it is echoed into logs.
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
    """Session cookie settings and lifetime from config/auth.json."""

    @app.before_request
    def _refresh_session_lifetime():
        if not session:
            return
        minutes = float(app.config["SST_SESSION_LIFETIME_MINUTES"])
        session.permanent = True
        session.modified = True
        app.permanent_session_lifetime = dt.timedelta(minutes=minutes)


def _register_csrf(app: Flask) -> None:
    """Protect cookie-authenticated writes, including the sign-in and sign-out forms."""

    def token() -> str:
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_urlsafe(32)
        return session["_csrf_token"]

    app.jinja_env.globals["csrf_token"] = token

    @app.before_request
    def _check_csrf():
        if not app.config["SST_CSRF_ENABLED"] or request.method in {"GET", "HEAD", "OPTIONS"}:
            return
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        expected = session.get("_csrf_token", "")
        if not expected or not hmac.compare_digest(supplied, expected):
            from src.errors import ApiError

            raise ApiError("forbidden", "This form expired. Refresh the page and try again.")


def _register_blueprints(app: Flask) -> None:
    from src.api.auth_api import bp as auth_api_bp
    from src.api.health import bp as health_bp
    from src.api import pages as pages_module
    from src.api.pages import bp as pages_bp

    app.register_blueprint(health_bp)
    # auth_api.py owns /api/auth/*; pages.py's auth_bp owns the HTML /login, /logout.
    app.register_blueprint(auth_api_bp)
    app.register_blueprint(pages_bp)
    # pages.py contributes five blueprints (auth, main, admin, models, event_visuals); all must
    # be registered or url_for calls in the templates fail.
    for extra in pages_module.BLUEPRINTS:
        if extra is not pages_bp:
            app.register_blueprint(extra)

    # Optional API slices are registered only when importable, and /api/health reports
    # any that are missing.
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
    # Refuse to start on broken configuration rather than discover it mid-demo.
    problems = store.validate()
    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(f"configuration is invalid, refusing to start: {joined}")
    set_store(store)
    app.config["SST_CONFIG_STORE"] = store
    return store


def _load_models(app: Flask, pipeline: Any | None) -> Any | None:
    """Resolve the analysis pipeline without ever failing the boot.

    Injected pipeline (tests), else one loaded from disk, else none: the server still runs and
    /api/health reports why analysis is unavailable.
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
        app.config["SST_PIPELINE_ERROR"] = (
            "Analysis models are unavailable. Ask an administrator to check the server "
            "logs and install the saved model artifacts."
        )
        return None

    set_pipeline(loaded)
    app.config["SST_PIPELINE"] = loaded
    app.config["SST_PIPELINE_ERROR"] = None
    logger.info(
        "analysis pipeline loaded: python=%s gtm=%s frontend_verified=%s",
        loaded.models.python.model_version,
        loaded.models.gtm.model_version,
        loaded.models.gtm.verified,
    )
    return loaded


def create_app(config: Mapping[str, Any] | None = None, **overrides: Any) -> Flask:
    """Build the application. ``config``/``overrides`` go over the defaults, e.g.
    create_app(TESTING=True, SST_DB_PATH=":memory:") for a test app with no models.
    """
    app = Flask(
        __name__,
        template_folder=str(REPO_ROOT / "templates"),
        static_folder=str(REPO_ROOT / "static"),
    )

    # Merge both ways of configuring the instance before the bootstrap reads them.
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
        "SST_CSRF_ENABLED": not supplied.get("TESTING", False),
        # Reverse proxies in front of the app (1 on Render or behind nginx). With 0,
        # X-Forwarded-For is ignored, because any client can send that header.
        "SST_TRUSTED_PROXY_HOPS": int(os.environ.get("SST_TRUSTED_PROXY_HOPS", "0")),
    }
    app.config.update(defaults)
    app.config.update(supplied)

    # The config store must exist before the cookie settings (from auth.json) are known.
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
        os.environ.get("SST_PRODUCTION") == "1" or
        store.auth_setting("session.cookie_secure", False)
    )
    app.config["PERMANENT_SESSION_LIFETIME"] = dt.timedelta(
        minutes=float(app.config["SST_SESSION_LIFETIME_MINUTES"])
    )
    app.config["REMEMBER_COOKIE_DURATION"] = dt.timedelta(
        days=float(app.config["SST_REMEMBER_DAYS"])
    )

    # Login rate limiters, shared by the form and the JSON API (limits from config/auth.json).
    from src.services.ratelimit import RateLimiter

    app.extensions["sst_limiter_login_ip"] = RateLimiter(
        limit=int(store.auth_setting("rate_limits.login_per_ip_per_minute", 20)),
        window_seconds=60.0,
    )
    app.extensions["sst_limiter_login_user"] = RateLimiter(
        limit=int(store.auth_setting("rate_limits.login_per_username_per_minute", 10)),
        window_seconds=60.0,
    )

    _configure_jinja(app, store)

    _configure_logging(app)

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

    from src.auth import _register_login_manager

    _register_login_manager(app)

    loaded = _load_models(app, app.config.get("SST_PIPELINE_INJECT"))

    # Pay librosa/numba and first-model-call costs now, not in the first request. AST's
    # first forward pass in particular is far slower than every later one.
    if app.config.get("SST_WARM_MODELS", True) and loaded is not None:
        try:
            loaded.warm()
        except Exception as exc:  # a warm-up failure must not stop the app from starting
            logger.warning("model warm-up failed at start-up: %s", exc)

    _apply_request_context(app)
    _register_csrf(app)
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


def _configure_jinja(app: Flask, store) -> None:
    """Register the fromjson filter and the template globals (severity scale, classes, model labels)
    every page needs.
    """
    import json as _json

    def _fromjson(value):
        # Macro payloads are already text; config values may arrive as real lists.
        if isinstance(value, (list, dict)):
            return value
        if value is None or value == "":
            return None
        return _json.loads(value)

    app.jinja_env.filters["fromjson"] = _fromjson

    def _review_reason(value):
        """'low_confidence,model_disagreement' -> the SRS phrases from manual_review_conditions.json."""
        if not value:
            return ""
        phrases = {c["id"]: c.get("srs_phrase", c["id"]) for c in store.review_conditions()}
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        if not all(part in phrases or part.replace("_", "").isalnum() for part in parts):
            return value  # free text, not a list of condition ids
        return ", ".join(phrases.get(part, part.replace("_", " ")) for part in parts)

    app.jinja_env.filters["review_reason"] = _review_reason

    def _when(value):
        """Naive-UTC datetime -> '26 Sep 2026, 12:36:44 UTC'; anything else unchanged."""
        return value.strftime("%d %b %Y, %H:%M:%S UTC") if hasattr(value, "strftime") else (value or "")

    app.jinja_env.filters["when"] = _when

    from src.models import (
        ALERT_STATUSES,
        CONSISTENCY_STATUSES,
        MODEL_LABELS,
        REVIEW_DECISIONS,
    )

    @app.context_processor
    def _inject_template_globals():  # pragma: no cover - exercised via page renders
        from src.app import APP_NAME, APP_VERSION  # local: avoid a circular import

        return {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "request_id": getattr(g, "request_id", None)
            or getattr(request, "request_id", None),
            "config": store,
            "thresholds": store.thresholds(),
            "severity_scale": store.severity_scale(),
            "severity_levels": store.severity_levels(),
            "class_names": store.class_names(),
            "critical_classes": sorted(store.critical_classes()),
            "quality_verdicts": store.quality_ordering(),
            "consistency_statuses": list(CONSISTENCY_STATUSES),
            "alert_statuses": list(ALERT_STATUSES),
            "review_decisions": list(REVIEW_DECISIONS),
            "review_conditions": store.review_conditions(),
            "model_labels": dict(MODEL_LABELS),
        }


def _secret_key(config: Mapping[str, Any]) -> str:
    """A stable secret key.

    In development it is derived from the repository path so restarts keep sessions (that key is
    predictable, so never use it in a deployment); with SST_PRODUCTION=1, SST_SECRET_KEY must be set.
    """
    configured = os.environ.get("SST_SECRET_KEY") or os.environ.get("SECRET_KEY")
    if configured:
        return configured
    if config.get("TESTING"):
        return "testing-secret-key"
    if os.environ.get("SST_PRODUCTION") == "1":
        raise RuntimeError("SST_SECRET_KEY must be set when SST_PRODUCTION=1")
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
