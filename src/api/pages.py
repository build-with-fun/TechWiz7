"""The server-rendered pages, and the one JSON endpoint the pages need.

Owner: sara.  SRS FR iv-ix (upload), FR lxiii-lxvii (dashboards, search), FR xxi-xxii
(the evidence visuals the detail page draws), plus the page-level half of the FR ii matrix.

Four blueprints live here, because the endpoint names are a contract the frontend builds its
``url_for`` calls on and a blueprint name is part of the endpoint:

======================  =====================  =====================================
Blueprint               Endpoints              Gate
======================  =====================  =====================================
``auth``                ``login``, ``logout``   public / signed in
``main``                ``index``, ``upload``, ``events``, ``event_detail``,
                        ``event_audio``, ``live``, ``alerts``, ``reviews``,
                        ``dashboard``, ``analytics``
``admin``               ``config``, ``users``   ``edit_config`` / ``manage_users``
``models``              ``list``                any signed-in user (read-only)
``event_visuals``       ``visuals``             owner, or ``download_any_audio``
======================  =====================  =====================================

Two rules hold throughout:

* **Every route is gated on a capability, never on a role string.** ``main.reviews`` asks for
  ``review_queue``; moving that capability is then a one-line change in ``src/auth.py``. A
  template that hides a link is presentation, not access control -- a reviewer typing the URL
  gets the same refusal as a reviewer clicking it.
* **A page that names a row the caller may not see answers 404, not 403.** FR lxvii: a normal
  user sees only their own events, and a 403 would confirm that someone else's event exists.
  A *capability* the caller's role can never hold is 403. That distinction is the whole of
  contract §1.4 and it is decided here, in ``_visible_event``.
"""

from __future__ import annotations

import datetime as dt
import logging

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required, login_user
from sqlalchemy import func, select
from jinja2 import TemplateNotFound

from src.auth import capability_required, owner_or_capability
from src.db import session_scope
from src.errors import ApiError
from src.models import Alert, Event, ModelVersion, Review, utcnow
from src.services.search import (
    filter_fields,
    parse_filters,
    run_search,
    describe_filters,
)

logger = logging.getLogger("sonicsentinel.api.pages")

#: FR lxii's statuses and the review decisions, for the list columns. Kept here rather than
#: in a template so the same words appear on the page and in the CSV export.
SEVERITY_TONE = {
    "Critical": "critical", "High": "high", "Medium": "medium",
    "Low": "low", "Informational": "info",
}

auth_bp = Blueprint("auth", __name__)
main_bp = Blueprint("main", __name__)
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")
models_bp = Blueprint("models", __name__, url_prefix="/models")
visuals_bp = Blueprint("event_visuals", __name__, url_prefix="/api")

#: Every blueprint this module contributes, in registration order. ``src.app`` registers them
#: all; ``bp`` below is kept as the module's headline blueprint for anything that imports it.
BLUEPRINTS = (auth_bp, main_bp, admin_bp, models_bp, visuals_bp)
bp = main_bp


# ---------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------


def _store():
    return current_app.config["SST_CONFIG_STORE"]


def _factory():
    return current_app.config["SST_SESSION_FACTORY"]


def _render(template: str, **context):
    """Render a page, turning a missing template into an honest, quiet failure.

    The templates are another engineer's deliverable and are landing in parallel, so this is
    a real state of the tree rather than a hypothetical. The user gets a page-shaped sentence
    and a request id; the log gets the template name, which is where it belongs -- a rendered
    error must never name a file on disk (FR lxxvii).
    """
    try:
        return render_template(template, **context)
    except TemplateNotFound:
        logger.error(
            "template %r is missing; the page cannot be rendered [%s]",
            template, getattr(request, "request_id", "-"),
        )
        raise ApiError(
            "internal_error",
            "This page is not available yet. It has been logged.",
        )


def _visible_event(session, event_id: int, *, for_download: bool = False) -> Event:
    """Fetch one event, or refuse in the way contract §1.4 requires.

    Missing, or belonging to someone else with no fleet-wide capability: **404**, so the
    request cannot be used to discover which ids exist. A caller who simply cannot perform
    this kind of action at all would be 403, and that is decided by the decorator before this
    function runs.
    """
    event = session.get(Event, event_id)
    if event is None:
        raise ApiError("not_found", "That event does not exist.")
    capability = "download_any_audio" if for_download else "view_all_events"
    if not current_user.can(capability) and event.created_by_id != current_user.id:
        raise ApiError("not_found", "That event does not exist.")
    return event


def _pagination(args, store) -> tuple[int, int]:
    """Page and size, from ``config/auth.json`` rather than literals."""
    default = int(store.auth_setting("pagination.default_page_size", 25)
                  or store.auth_setting("pagination.default", 25) or 25)
    maximum = int(store.auth_setting("pagination.max_page_size", 200)
                  or store.auth_setting("pagination.max", 200) or 200)
    return default, maximum


def _search_context(viewer, *, default_statuses=None) -> dict:
    """Everything the search page needs, built once so the page and the API agree."""
    store = _store()
    default_size, max_size = _pagination(request.args, store)
    filters = parse_filters(request.args, store, viewer=viewer,
                            default_page_size=default_size, max_page_size=max_size)
    with session_scope(_factory()) as session:
        rows, meta = run_search(session, filters, store)
        for row in rows:
            session.expunge(row)
    return {
        "events": rows,
        "meta": meta,
        "filters": filters,
        "problems": list(filters.problems),
        "filter_fields": filter_fields(store),
        "filter_summary": describe_filters(filters),
        "severity_tone": SEVERITY_TONE,
    }


# ---------------------------------------------------------------------------------------
# auth -- the HTML sign-in page.  The JSON endpoints are in src/api/auth_api.py
# ---------------------------------------------------------------------------------------


@auth_bp.get("/login")
def login():
    """The sign-in page.

    A signed-in user who lands here is sent on rather than shown a second sign-in form --
    except when they asked to switch account, which is how the demo shows two roles in one
    session without clearing cookies by hand.
    """
    if current_user.is_authenticated and request.args.get("switch") != "1":
        return redirect(_home_for(current_user))
    return _render(
        "auth/login.html",
        next=request.args.get("next") or "",
        error=None,
        switch=request.args.get("switch") == "1",
    )


@auth_bp.post("/logout")
@login_required
def logout():
    """Sign out from a page. A form posts here; the JSON client uses ``POST /api/auth/logout``."""
    from src.auth import audit_login, sign_out
    from flask import g

    username, role, user_id = current_user.username, current_user.role, current_user.id
    with session_scope(_factory()) as session:
        audit_login(session, action="logout", username=username, role=role, user_id=user_id,
                    request_id=getattr(g, "request_id", None), detail="signed out from a page")
    sign_out()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))


def _home_for(user) -> str:
    """Where a user lands after signing in: the most useful page their role can open.

    Deliberately capability-driven. If the navigation is ever re-cut, this keeps working
    rather than sending a security operator to a dashboard they cannot see.
    """
    for capability, endpoint in (
        ("view_dashboards", "main.dashboard"),
        ("view_alerts", "main.alerts"),
        ("review_queue", "main.reviews"),
    ):
        if user.can(capability):
            return url_for(endpoint)
    return url_for("main.events")


# ---------------------------------------------------------------------------------------
# main -- the pages every signed-in user shares
# ---------------------------------------------------------------------------------------


@main_bp.get("/")
@login_required
def index():
    """The landing page: a way in, chosen by what the role can actually do."""
    return redirect(_home_for(current_user))


@main_bp.get("/upload")
@capability_required("upload_audio")
def upload():
    """FR iv-ix: the upload page.

    The accepted formats and the size limit come from ``config/auth.json`` and the app config,
    so the page states the same limit the server enforces. A page that promises a format the
    server refuses is worse than no page.
    """
    store = _store()
    return _render(
        "upload.html",
        accepted=store.auth_setting("uploads.accepted_content_types", []) or [],
        max_bytes=current_app.config.get("MAX_CONTENT_LENGTH"),
        max_mb=round((current_app.config.get("MAX_CONTENT_LENGTH") or 0) / (1024 * 1024), 1),
        quality_verdicts=list(store.quality_ordering()),
        analysis_ready=bool(current_app.config.get("SST_PIPELINE")),
        pipeline_error=current_app.config.get("SST_PIPELINE_ERROR"),
    )


@main_bp.get("/events")
@capability_required("view_own_events")
def events():
    """FR lxvii: browse and search. A normal user's search is scoped inside the query."""
    context = _search_context(current_user)
    context["page_title"] = "Events"
    return _render("events.html", **context)


@main_bp.get("/events/<int:event_id>")
@capability_required("view_own_events")
def event_detail(event_id: int):
    """One event, in full: both models' classes and confidences, the comparison, the quality
    verdict, the severity and the rule that fired -- FR xxxi-xxxvi, FR xl, FR lxii."""
    store = _store()
    with session_scope(_factory()) as session:
        event = _visible_event(session, event_id)
        scores: dict[str, list] = {}
        for score in event.confidence_scores:
            scores.setdefault(score.model_name, []).append(score)
        for rows in scores.values():
            rows.sort(key=lambda s: (s.rank if s.rank is not None else 99))
        alerts = sorted(event.alerts, key=lambda a: (a.created_at or utcnow()), reverse=True)
        reviews = sorted(event.reviews, key=lambda r: (r.queued_at or utcnow()), reverse=True)
        payload = {
            "event": event,
            "audio": event.audio_file,
            "scores": scores,
            "alerts": alerts,
            "reviews": reviews,
            "severity_tone": SEVERITY_TONE,
            "rule": (event.config_snapshot or {}).get("alert_rule") if event.config_snapshot else None,
        }
        for name in ("event", "audio"):
            if payload[name] is not None:
                session.expunge(payload[name])
        for row in alerts + reviews:
            session.expunge(row)
        for rows in scores.values():
            for row in rows:
                session.expunge(row)
    payload["page_title"] = f"Event {event_id}"
    return _render("event_detail.html", **payload)


@main_bp.get("/events/<int:event_id>/audio")
@owner_or_capability("download_any_audio")
def event_audio(event_id: int):
    """Stream the stored recording (FR lxxi: audio is stored on disk, not in the database).

    ``as_attachment`` is off by default so the detail page can play it inline; the download
    button asks for ``?download=1``.
    """
    from flask import send_from_directory

    layout = current_app.config["SST_STORAGE"]
    with session_scope(_factory()) as session:
        event = _visible_event(session, event_id, for_download=True)
        audio = event.audio_file
        if audio is None:
            raise ApiError("not_found", "That event has no stored recording.")
        stored_path, filename = audio.stored_path, audio.filename
    try:
        target = layout.resolve(stored_path)
    except ValueError:
        # A path that escapes the storage root is not a missing file, it is a bad record.
        logger.error("stored path for event %s escapes the storage root", event_id)
        raise ApiError("storage_error", "That recording could not be located.")
    if not target.exists():
        # The recording is gone -- expired by the FR lxxx retention policy, or moved. The
        # event's analysis is still intact and still auditable, and saying so is the point.
        raise ApiError(
            "no_audio",
            "That recording is no longer on disk. Its analysis and history are unaffected.",
        )
    return send_from_directory(
        target.parent, target.name,
        as_attachment=request.args.get("download") == "1",
        download_name=filename,
    )


@main_bp.get("/live")
@capability_required("live_session")
def live():
    """FR xxxvi: the live microphone monitor."""
    store = _store()
    return _render(
        "live.html",
        analysis_ready=bool(current_app.config.get("SST_PIPELINE")),
        pipeline_error=current_app.config.get("SST_PIPELINE_ERROR"),
        window_seconds=list(store.thresholds().get("live_window_seconds", [1, 3]) or [1, 3]),
        classes=store.class_names(),
    )


@main_bp.get("/alerts")
@capability_required("view_alerts")
def alerts():
    """FR liv-lvi: the alert console."""
    store = _store()
    status = request.args.get("status") or "Open"
    severity = request.args.getlist("severity")
    with session_scope(_factory()) as session:
        statement = select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc()).limit(200)
        if status and status != "all":
            statement = statement.where(Alert.status == status)
        if severity:
            statement = statement.where(Alert.severity.in_(severity))
        rows = session.execute(statement).scalars().all()
        for row in rows:
            session.expunge(row)
        counts = dict(
            session.execute(
                select(Alert.status, func.count(Alert.id)).group_by(Alert.status)
            ).all()
        )
    return _render(
        "alerts.html",
        alerts=rows,
        counts=counts,
        current_status=status,
        severities=list(store.severity_scale()),
        severity_tone=SEVERITY_TONE,
        can_acknowledge=current_user.can("acknowledge_alerts"),
        page_title="Alerts",
    )


@main_bp.get("/reviews")
@capability_required("review_queue")
def reviews():
    """FR lvii-lxi: the manual-review queue, oldest and most severe first."""
    store = _store()
    status = request.args.get("status") or "Queued"
    with session_scope(_factory()) as session:
        statement = (
            select(Review)
            .order_by(Review.priority.asc().nullslast(), Review.queued_at.asc())
            .limit(200)
        )
        if status and status != "all":
            statement = statement.where(Review.status == status)
        rows = session.execute(statement).scalars().all()
        for row in rows:
            session.expunge(row)
        counts = dict(
            session.execute(
                select(Review.status, func.count(Review.id)).group_by(Review.status)
            ).all()
        )
    return _render(
        "reviews.html",
        reviews=rows,
        counts=counts,
        current_status=status,
        conditions=store.review_conditions(),
        can_decide=current_user.can("review_decide"),
        page_title="Manual review",
    )


@main_bp.get("/dashboard")
@capability_required("view_dashboards")
def dashboard():
    """FR lxiii-lxv: the operational dashboards."""
    store = _store()
    now = utcnow()
    day_ago = now - dt.timedelta(hours=24)
    week_ago = now - dt.timedelta(days=7)
    with session_scope(_factory()) as session:
        by_class = session.execute(
            select(Event.predicted_class, func.count(Event.id))
            .where(Event.created_at >= week_ago)
            .group_by(Event.predicted_class)
        ).all()
        by_severity = session.execute(
            select(Event.severity, func.count(Event.id))
            .where(Event.created_at >= week_ago)
            .group_by(Event.severity)
        ).all()
        by_status = session.execute(
            select(Event.status, func.count(Event.id)).group_by(Event.status)
        ).all()
        totals = {
            "events_24h": session.execute(
                select(func.count(Event.id)).where(Event.created_at >= day_ago)
            ).scalar() or 0,
            "events_7d": session.execute(
                select(func.count(Event.id)).where(Event.created_at >= week_ago)
            ).scalar() or 0,
            "events_total": session.execute(select(func.count(Event.id))).scalar() or 0,
            "open_alerts": session.execute(
                select(func.count(Alert.id)).where(Alert.status == "Open")
            ).scalar() or 0,
            "awaiting_review": session.execute(
                select(func.count(Event.id)).where(Event.requires_manual_review.is_(True))
            ).scalar() or 0,
            "critical_7d": session.execute(
                select(func.count(Event.id))
                .where(Event.created_at >= week_ago, Event.severity == "Critical")
            ).scalar() or 0,
        }
    return _render(
        "dashboard.html",
        totals=totals,
        by_class=[{"class": c, "count": n} for c, n in by_class],
        by_severity=[{"severity": s, "count": n} for s, n in by_severity],
        by_status=[{"status": s, "count": n} for s, n in by_status],
        severity_scale=list(store.severity_scale()),
        critical_classes=sorted(store.critical_classes()),
        severity_tone=SEVERITY_TONE,
        page_title="Dashboard",
    )


@main_bp.get("/analytics")
@capability_required("view_analytics")
def analytics():
    """FR lxviii: the aggregate queries, with the window the caller asked for.

    The window is a query parameter rather than a fixed "last 7 days", because an evaluator
    will ask "and the last hour?" during the demo.
    """
    store = _store()
    hours = request.args.get("hours", type=int) or 24 * 7
    hours = max(1, min(hours, 24 * 365))
    since = utcnow() - dt.timedelta(hours=hours)
    with session_scope(_factory()) as session:
        rows = session.execute(
            select(
                Event.predicted_class,
                func.count(Event.id),
                func.avg(Event.top_confidence),
                func.avg(Event.confidence_difference),
            )
            .where(Event.created_at >= since)
            .group_by(Event.predicted_class)
        ).all()
        consistency = session.execute(
            select(Event.consistency_status, func.count(Event.id))
            .where(Event.created_at >= since)
            .group_by(Event.consistency_status)
        ).all()
        quality = session.execute(
            select(Event.quality_verdict, func.count(Event.id))
            .where(Event.created_at >= since)
            .group_by(Event.quality_verdict)
        ).all()
    return _render(
        "analytics.html",
        hours=hours,
        since=since,
        by_class=[
            {
                "class": name,
                "count": count,
                "mean_confidence": round(float(avg_conf or 0), 4),
                "mean_difference": round(float(avg_diff or 0), 4),
            }
            for name, count, avg_conf, avg_diff in rows
        ],
        by_consistency=[{"status": s, "count": n} for s, n in consistency],
        by_quality=[{"verdict": q, "count": n} for q, n in quality],
        consistency_statuses=store.consistency_ordering()
        if hasattr(store, "consistency_ordering") else [],
        severity_tone=SEVERITY_TONE,
        page_title="Analytics",
    )


# ---------------------------------------------------------------------------------------
# admin -- configuration and users
# ---------------------------------------------------------------------------------------


@admin_bp.get("/config")
@capability_required("edit_config")
def config():
    """FR liii / FR lxxx: the live-editable configuration.

    Read-only to look at; the write path is ``PUT /api/admin/config/<file>`` in the admin API
    slice, which runs the same validator as boot before anything touches the disk.
    """
    store = _store()
    snapshot = store.snapshot()
    return _render(
        "admin/config.html",
        config_dir=str(store.config_dir),
        alert_rules_dir=str(store.alert_rules_dir),
        snapshot=snapshot,
        hashes=snapshot.to_dict().get("hashes") if hasattr(snapshot, "to_dict") else None,
        thresholds=store.thresholds(),
        classes=store.classes_config(),
        severity_scale=list(store.severity_scale()),
        severity_levels=store.severity_levels(),
        effective_rules=store.effective_rules(),
        review_conditions=store.review_conditions(),
        retention=store.retention(),
        can_edit=True,
        page_title="Configuration",
    )


@admin_bp.get("/users")
@capability_required("manage_users")
def users():
    """FR i-ii: accounts and roles."""
    from src.models import ROLES, ROLE_LABELS, User

    with session_scope(_factory()) as session:
        rows = session.execute(select(User).order_by(User.role, User.username)).scalars().all()
        for row in rows:
            session.expunge(row)
        counts = dict(
            session.execute(select(User.role, func.count(User.id)).group_by(User.role)).all()
        )
    return _render(
        "admin/users.html",
        users=rows,
        role_counts=counts,
        roles=[{"value": r, "label": ROLE_LABELS.get(r, r)} for r in ROLES],
        page_title="Users",
    )


@models_bp.get("/")
@login_required
def list():
    """FR lxxv: which models are in service, and which is active.

    Open to any signed-in user on purpose: a person reading an event's result is entitled to
    know which model version produced it. Registering and activating is what needs
    ``manage_models``, and those are write endpoints in the admin API slice.
    """
    with session_scope(_factory()) as session:
        rows = session.execute(
            select(ModelVersion).order_by(ModelVersion.model_name, ModelVersion.registered_at.desc())
        ).scalars().all()
        for row in rows:
            session.expunge(row)
    return _render(
        "models.html",
        versions=rows,
        can_manage=current_user.can("manage_models"),
        page_title="Models",
    )


# ---------------------------------------------------------------------------------------
# event_visuals -- the waveform and spectrogram the detail page draws (FR xxi-xxii)
# ---------------------------------------------------------------------------------------


@visuals_bp.get("/events/<int:event_id>/visuals")
@owner_or_capability("download_any_audio")
def visuals(event_id: int):
    """Peaks and a spectrogram for one stored recording, downsampled to stay small.

    Computed server-side and cached, so the browser receives a bounded payload instead of
    decoding an audio file itself -- and so what the page draws is the same signal the models
    were given, not a second, independent decoding of it.
    """
    from src.services.visuals import build_visuals

    with session_scope(_factory()) as session:
        event = _visible_event(session, event_id, for_download=True)
        audio = event.audio_file
        if audio is None:
            raise ApiError("not_found", "That event has no stored recording.")
        stored_path = audio.stored_path
        duration = audio.duration_sec
        sample_rate = audio.sample_rate

    payload = build_visuals(
        current_app.config["SST_STORAGE"],
        stored_path,
        event_id=event_id,
        fallback_duration=duration,
        fallback_sample_rate=sample_rate,
    )
    return jsonify(payload)
