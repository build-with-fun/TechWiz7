"""Server-rendered pages, plus the visuals JSON the event page loads.

Blueprints: auth, main, admin, models and event_visuals. Routes check capabilities from
src/auth.py rather than role names. Events the user can't see return 404, not 403.
"""

from __future__ import annotations

import datetime as dt
import logging

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload
from jinja2 import TemplateNotFound

from src.auth import (
    audit_login,
    capability_required,
    has_capability,
    owner_or_capability,
)
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

# FR lxii statuses and review decisions, shared by the list pages and the CSV export.
SEVERITY_TONE = {
    "Critical": "critical", "High": "high", "Medium": "medium",
    "Low": "low", "Informational": "info",
}

auth_bp = Blueprint("auth", __name__)
main_bp = Blueprint("main", __name__)
admin_bp = Blueprint("admin", __name__, url_prefix="/admin")
models_bp = Blueprint("models", __name__, url_prefix="/models")
visuals_bp = Blueprint("event_visuals", __name__, url_prefix="/api")

# Registered by src.app in this order.
BLUEPRINTS = (auth_bp, main_bp, admin_bp, models_bp, visuals_bp)
bp = main_bp




def _store():
    return current_app.config["SST_CONFIG_STORE"]


def _factory():
    return current_app.config["SST_SESSION_FACTORY"]


def _render(template: str, **context):
    """Render a template. A missing template shows a generic error page and is logged."""
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
    """Fetch one event, or 404 if it doesn't exist or the user can't see it."""
    event = session.get(Event, event_id)
    if event is None:
        raise ApiError("not_found", "That event does not exist.")
    capability = "download_any_audio" if for_download else "view_all_events"
    if not current_user.can(capability) and event.created_by_id != current_user.id:
        raise ApiError("not_found", "That event does not exist.")
    return event


def _pagination(args, store) -> tuple[int, int]:
    """Page and page size, bounded by config/auth.json."""
    default = int(store.auth_setting("pagination.default_page_size", 25)
                  or store.auth_setting("pagination.default", 25) or 25)
    maximum = int(store.auth_setting("pagination.max_page_size", 200)
                  or store.auth_setting("pagination.max", 200) or 200)
    return default, maximum


def _event_id_int(value: str) -> int:
    """Parse the event id; base.html uses an __EVENT_ID__ placeholder in some URLs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ApiError("bad_request", "invalid event id")


def _search_context(viewer, *, default_statuses=None) -> dict:
    """Search page context, shared with the API."""
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




@auth_bp.get("/login")
def login():
    """Sign-in page. Signed-in users are redirected unless they want to switch account."""
    if current_user.is_authenticated and request.args.get("switch") != "1":
        return redirect(_home_for(current_user))
    return _render(
        "auth/login.html",
        next=request.args.get("next") or "",
        error=None,
        switch=request.args.get("switch") == "1",
    )


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Create a normal user account. Other roles are assigned by an administrator."""
    from src.auth import client_ip, hash_password, password_problems, sign_in
    from src.db import record_audit
    from src.models import User

    if request.method == "GET":
        return _render("auth/register.html", error=None)

    username = str(request.form.get("username", "")).strip()
    email = str(request.form.get("email", "")).strip().lower()
    display_name = str(request.form.get("display_name", "")).strip()
    password = request.form.get("password", "")
    if not (3 <= len(username) <= 64) or not all(c.isalnum() or c in "_-" for c in username):
        return _render("auth/register.html", error="Use 3–64 letters, numbers, underscores or hyphens for the username."), 400
    if not (3 <= len(email) <= 255) or email.count("@") != 1 or "." not in email.rsplit("@", 1)[-1]:
        return _render("auth/register.html", error="Enter a valid email address."), 400
    if len(display_name) > 128:
        return _render("auth/register.html", error="Display name must be 128 characters or fewer."), 400
    problems = password_problems(password, _store())
    if problems:
        return _render("auth/register.html", error="Password " + "; ".join(problems) + "."), 400

    try:
        with session_scope(_factory()) as session:
            user = User(username=username, email=email, display_name=display_name or username,
                        password_hash=hash_password(password), role="normal_user")
            session.add(user)
            session.flush()
            record_audit(session, action="user_create", actor=user, target_type="user",
                         target_id=str(user.id), detail="self-registered normal-user account",
                         ip_address=client_ip())
            session.expunge(user)
    except IntegrityError:
        return _render("auth/register.html", error="That username or email is already in use."), 409
    sign_in(user)
    return redirect(url_for("main.index"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Edit your own contact and display details."""
    from src.db import record_audit
    from src.models import User

    if request.method == "POST":
        display_name = str(request.form.get("display_name", "")).strip()
        email = str(request.form.get("email", "")).strip().lower()
        if len(display_name) > 128 or not display_name:
            return _render("auth/profile.html", error="Enter a display name of up to 128 characters."), 400
        if not (3 <= len(email) <= 255) or email.count("@") != 1 or "." not in email.rsplit("@", 1)[-1]:
            return _render("auth/profile.html", error="Enter a valid email address."), 400
        try:
            with session_scope(_factory()) as session:
                row = session.get(User, current_user.id)
                before = {"display_name": row.display_name, "email": row.email}
                row.display_name = display_name
                row.email = email
                record_audit(session, action="user_update", actor=row, target_type="user",
                             target_id=str(row.id), before=before,
                             after={"display_name": display_name, "email": email})
        except IntegrityError:
            return _render("auth/profile.html", error="That email address is already in use."), 409
        flash("Profile updated.", "success")
        return redirect(url_for("auth.profile"))
    return _render("auth/profile.html", error=None)


@auth_bp.post("/session")
def session_login():
    """Form sign-in. Uses the same src.auth.authenticate as POST /api/auth/login.

    ``next`` is only followed if it is a relative path, to avoid an open redirect.
    """
    from src.auth import authenticate, client_ip, sign_in
    from flask import g

    username = str(request.form.get("username", "")).strip()
    password = request.form.get("password", "")
    remember = request.form.get("remember") == "1"

    def _fail(message: str):
        return _render(
            "auth/login.html",
            next=request.form.get("next") or "",
            error=message,
            switch=False,
        ), 401 if message != "Too many sign-in attempts. Try again shortly." else 429

    if not username or not password:
        return _fail("Enter both a username and a password.")

    factory = _factory()
    request_id = getattr(g, "request_id", None)
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
            return _fail(f"Too many sign-in attempts from {label}. Try again in "
                         f"{result.retry_after_seconds} seconds.")

    with session_scope(factory) as s:
        outcome = authenticate(s, username, password, store=_store())
        if outcome.ok:
            row = outcome.user
            audit_login(s, action="login_success", username=row.username, role=row.role,
                        user_id=row.id, request_id=request_id,
                        detail=f"signed in from {client_ip()} (form)")
            s.expunge(row)
        else:
            audit_login(s, action="login_failure", username=username, outcome="failure",
                        detail=outcome.code, request_id=request_id)

    if not outcome.ok:
        # Don't reveal whether the username exists.
        return _fail("Sign-in failed. Check the username and password, "
                     "or the account may be locked or disabled.")

    limiter_user.reset(username.lower())
    sign_in(outcome.user, remember=remember)

    next_url = request.form.get("next") or ""
    if next_url.startswith("/") and not next_url.startswith("//"):
        return redirect(next_url)
    return redirect(_home_for(outcome.user))


@auth_bp.post("/logout")
@login_required
def logout():
    """Sign out from a form (the API uses POST /api/auth/logout)."""
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
    """Where to send a user after sign-in, based on their capabilities."""
    for capability, endpoint in (
        ("view_dashboards", "main.dashboard"),
        ("view_alerts", "main.alerts"),
        ("review_queue", "main.reviews"),
    ):
        # Works for both the User row and the Flask-Login wrapper.
        if has_capability(getattr(user, "role", None), capability):
            return url_for(endpoint)
    return url_for("main.events")




def _landing_metrics() -> dict:
    """Test-set results of both served models, read from their metrics files."""
    import json

    from src.db import REPO_ROOT

    out: dict = {}
    sources = (
        ("python", REPO_ROOT / "python_models" / "best" / "model_meta.json",
         lambda d: d.get("metrics", {}), "critical_recall"),
        ("gtm", REPO_ROOT / "gtm_model" / "gtm_metrics.json", lambda d: d, "critical_macro_recall"),
    )
    for key, path, pick, critical_key in sources:
        try:
            doc = pick(json.loads(path.read_text(encoding="utf-8")))
            out[key] = {"accuracy": float(doc["accuracy"]), "macro_f1": float(doc["macro_f1"]),
                        "critical_recall": float(doc[critical_key])}
        except (OSError, ValueError, KeyError, TypeError):
            out[key] = None
    return out


@main_bp.get("/")
def index():
    """Signed-in users go to their home page; everyone else sees the public page."""
    if current_user.is_authenticated:
        return redirect(_home_for(current_user))
    return _render("landing.html", metrics=_landing_metrics())


@main_bp.get("/upload")
@capability_required("upload_audio")
def upload():
    """FR iv-ix: upload page."""
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
    """FR lxvii: event search. Normal users only see their own events."""
    context = _search_context(current_user)
    context["page_title"] = "Events"
    return _render("events.html", **context)


@main_bp.get("/events/<event_id>")
@capability_required("view_own_events")
def event_detail(event_id: str):
    """Event detail page.

    The id is a string in the route because base.html builds URLs with an __EVENT_ID__
    placeholder.
    """
    event_id = _event_id_int(event_id)
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
            row = payload[name]
            if row is not None and row in session:
                session.expunge(row)
        for row in alerts + reviews:
            if row in session:
                session.expunge(row)
        for rows in scores.values():
            for row in rows:
                if row in session:
                    session.expunge(row)
    payload["page_title"] = f"Event {event_id}"
    return _render("event_detail.html", **payload)


@main_bp.get("/events/<event_id>/audio")
@owner_or_capability("download_any_audio")
def event_audio(event_id: str):
    """Stream the stored recording. ``?download=1`` sends it as an attachment."""
    event_id = _event_id_int(event_id)
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
        # A path outside the storage root means a bad record.
        logger.error("stored path for event %s escapes the storage root", event_id)
        raise ApiError("storage_error", "That recording could not be located.")
    if not stored_path or not target.is_file():
        # The file was purged or moved; the event itself is still there.
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
    """Live microphone page."""
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
    """FR liv-lvi: alert console."""
    store = _store()
    status = request.args.get("status") or "Open"
    severity = request.args.getlist("severity")
    with session_scope(_factory()) as session:
        # joinedload because the rows are expunged and the template reads alert.event.
        statement = (
            select(Alert)
            .options(joinedload(Alert.event))
            .order_by(Alert.created_at.desc(), Alert.id.desc())
            .limit(200)
        )
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
    """FR lvii-lxi: manual-review queue, oldest and most severe first."""
    store = _store()
    status = request.args.get("status") or "Pending Review"
    with session_scope(_factory()) as session:
        # joinedload, same as alerts().
        statement = (
            select(Review)
            .options(joinedload(Review.event))
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


def _smooth_chart(values: list[int], width: float = 320.0, height: float = 150.0,
                  pad: float = 14.0) -> dict:
    """SVG path for a smooth line (Catmull-Rom as cubic Beziers) and the area under it."""
    n = len(values)
    top = max(values + [1])
    xs = [pad + i * (width - 2 * pad) / max(n - 1, 1) for i in range(n)]
    ys = [height - pad - (v / top) * (height - 2 * pad) for v in values]
    points = list(zip(xs, ys))
    if not points:
        return {"line": "", "area": "", "points": [], "peak": None}
    line = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for i in range(n - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < n else p2
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        line += (f" C{c1[0]:.1f},{max(c1[1], pad / 2):.1f} {c2[0]:.1f},{max(c2[1], pad / 2):.1f}"
                 f" {p2[0]:.1f},{p2[1]:.1f}")
    area = f"{line} L{xs[-1]:.1f},{height:.1f} L{xs[0]:.1f},{height:.1f} Z"
    peak = max(range(n), key=lambda i: values[i])
    return {"line": line, "area": area, "points": [{"x": x, "y": y} for x, y in points],
            "peak": {"x": xs[peak], "y": ys[peak], "value": values[peak], "index": peak},
            "width": width, "height": height}


@main_bp.get("/dashboard")
@capability_required("view_own_events")
def dashboard():
    """Dashboard (FR lxiii), plus the administrator block (FR lxv).

    Normal users see their own events. Roles with analytics access also get averages,
    disagreements, poor-quality counts and a 14-day trend.
    """
    from src.models import AudioFile
    from src.services.monitoring import compute_anomalies

    store = _store()
    now = utcnow()
    day_ago = now - dt.timedelta(hours=24)
    week_ago = now - dt.timedelta(days=7)
    fortnight_ago = (now - dt.timedelta(days=13)).replace(hour=0, minute=0, second=0, microsecond=0)
    see_all = current_user.can("view_all_events")

    def scoped(query):
        return query if see_all else query.where(Event.created_by_id == current_user.id)

    def count(*where):
        return session.execute(scoped(select(func.count(Event.id)).where(*where))).scalar() or 0

    with session_scope(_factory()) as session:
        by_class = session.execute(scoped(
            select(Event.predicted_class, func.count(Event.id))
            .where(Event.created_at >= week_ago).group_by(Event.predicted_class))).all()
        by_severity = session.execute(scoped(
            select(Event.severity, func.count(Event.id))
            .where(Event.created_at >= week_ago).group_by(Event.severity))).all()
        by_status = session.execute(scoped(
            select(Event.status, func.count(Event.id)).group_by(Event.status))).all()
        totals = {
            "events_24h": count(Event.created_at >= day_ago),
            "events_7d": count(Event.created_at >= week_ago),
            "events_total": count(),
            "open_alerts": session.execute(scoped(
                select(func.count(Alert.id)).join(Event, Alert.event_id == Event.id)
                .where(Alert.status == "Open"))).scalar() or 0,
            "awaiting_review": count(Event.requires_manual_review.is_(True),
                                     Event.status != "Reviewed", Event.status != "Closed"),
            "critical_7d": count(Event.created_at >= week_ago, Event.severity == "Critical"),
            "quality_warnings_7d": count(Event.created_at >= week_ago,
                                         Event.quality_verdict.in_(("Poor", "Unusable"))),
        }
        recent_rows = session.execute(scoped(
            select(Event, AudioFile.filename).join(AudioFile, Event.audio_file_id == AudioFile.id)
            .order_by(Event.created_at.desc()).limit(8))).all()
        recent = [{"id": e.id, "filename": name, "created_at": e.created_at,
                   "predicted_class": e.predicted_class, "confidence": e.top_confidence,
                   "quality": e.quality_verdict, "severity": e.severity, "status": e.status,
                   "consistency": e.consistency_status, "review": e.requires_manual_review}
                  for e, name in recent_rows]
        critical_recent = [r for r in recent if r["severity"] == "Critical"]
        # SRS Step 18: latest detection with both models.
        latest = None
        if recent_rows:
            event, name = recent_rows[0]
            top3: dict[str, list] = {"python": [], "gtm": []}
            for score in sorted(event.confidence_scores, key=lambda sc: sc.rank if sc.rank is not None else 99):
                if score.model_name in top3 and len(top3[score.model_name]) < 3:
                    top3[score.model_name].append({"class": score.class_name, "confidence": score.confidence})
            last_alert = max(event.alerts, key=lambda a: a.created_at or utcnow(), default=None)
            latest = {"id": event.id, "filename": name, "created_at": event.created_at,
                      "source": event.source, "final_class": event.final_class or event.predicted_class,
                      "confidence": event.top_confidence, "difference": event.confidence_difference,
                      "quality": event.quality_verdict, "severity": event.severity, "status": event.status,
                      "consistency": event.consistency_status, "review": event.requires_manual_review,
                      "python": top3["python"], "gtm": top3["gtm"],
                      "alert_status": last_alert.status if last_alert else None,
                      "has_audio": bool(event.audio_file and event.audio_file.stored_path)}
        review_rows = session.execute(scoped(
            select(Event).where(Event.requires_manual_review.is_(True),
                                Event.status.notin_(("Reviewed", "Closed")))
            .order_by(Event.created_at.desc()).limit(5))).scalars().all()
        review_items = [{"id": e.id, "class": e.final_class or e.predicted_class, "at": e.created_at,
                         "reason": e.review_reason, "severity": e.severity} for e in review_rows]
        # FR lxvi: high and critical events, newest first.
        timeline_rows = session.execute(scoped(
            select(Event).where(Event.created_at >= week_ago,
                                Event.severity.in_(("High", "Critical")))
            .order_by(Event.created_at.desc()).limit(12))).scalars().all()
        timeline = [{"id": e.id, "at": e.created_at, "class": e.predicted_class,
                     "severity": e.severity, "status": e.status,
                     "review": e.requires_manual_review} for e in timeline_rows]
        day_col = func.date(Event.created_at)
        activity_rows = dict(session.execute(scoped(
            select(day_col, func.count(Event.id)).where(Event.created_at >= fortnight_ago)
            .group_by(day_col))).all())
        activity_days = [(fortnight_ago + dt.timedelta(days=offset)).date() for offset in range(14)]
        activity_counts = [int(activity_rows.get(d.isoformat(), 0)) for d in activity_days]
        activity = {"days": activity_days, "counts": activity_counts, "total": sum(activity_counts),
                    "chart": _smooth_chart(activity_counts)}
        status_counts = {status: int(n) for status, n in by_status}
        finished = status_counts.get("Reviewed", 0) + status_counts.get("Closed", 0)
        needing = finished + totals["awaiting_review"]
        review_progress = {"done": finished, "total": needing,
                           "pct": round(100 * finished / needing) if needing else 100}
        in_service = []
        for row in session.execute(select(ModelVersion).where(ModelVersion.is_active.is_(True))
                                   .order_by(ModelVersion.model_name)).scalars():
            metrics = row.metrics or {}
            test = metrics.get("accuracy") if metrics.get("split") == "test" else None
            in_service.append({"name": "Python model" if row.model_name == "python" else "Teachable Machine",
                               "version": row.version, "algorithm": row.algorithm or row.label or "",
                               "test_accuracy": test})
        admin = None
        if current_user.can("view_analytics"):
            avg_conf = session.execute(select(func.avg(Event.top_confidence))
                                       .where(Event.created_at >= week_ago)).scalar()
            day = func.date(Event.created_at)
            trend_rows = dict(session.execute(select(day, func.count(Event.id))
                                              .where(Event.created_at >= fortnight_ago)
                                              .group_by(day)).all())
            trend = []
            for offset in range(14):
                d = (fortnight_ago + dt.timedelta(days=offset)).date().isoformat()
                trend.append({"day": d, "count": int(trend_rows.get(d, 0))})
            admin = {
                "average_confidence": avg_conf,
                "disagreements_7d": count(Event.created_at >= week_ago,
                                          Event.consistency_status == "Model Disagreement"),
                "poor_quality_7d": totals["quality_warnings_7d"],
                "critical_alerts_7d": session.execute(select(func.count(Alert.id)).where(
                    Alert.created_at >= week_ago, Alert.severity == "Critical")).scalar() or 0,
                "trend": trend,
                "trend_max": max([t["count"] for t in trend] + [1]),
                "anomalies": (compute_anomalies(session, store.thresholds())["anomalies"]
                              if current_user.can("read_audit") else None),
            }
    return _render(
        "dashboard.html",
        activity=activity,
        review_progress=review_progress,
        totals=totals,
        in_service=in_service,
        latest=latest,
        review_items=review_items,
        recent=recent,
        critical_recent=critical_recent,
        timeline=timeline,
        admin=admin,
        scope_all=see_all,
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
    """FR lxviii analytics for the last ``?hours=`` (default one week)."""
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
        confidences = [c for (c,) in session.execute(
            select(Event.top_confidence).where(Event.created_at >= since,
                                               Event.top_confidence.is_not(None))).all()]
        decided = session.execute(
            select(Review.original_python_class, Review.final_class, Review.false_alarm)
            .where(Review.decided_at >= since, Review.decision != "pending")).all()
        alert_rows = session.execute(
            select(Alert.status, Alert.severity, Alert.created_at, Alert.acknowledged_at,
                   Alert.resolved_at, Alert.is_false_alarm)
            .where(Alert.created_at >= since)).all()
    critical = set(store.critical_classes())
    errors = _review_errors(decided, critical)
    return _render(
        "analytics.html",
        confidence_histogram=_histogram(confidences),
        review_errors=errors,
        alert_response=_alert_response(alert_rows),
        critical_frequency=sum(r[1] for r in rows if r[0] in critical),
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


def _histogram(values, bins: int = 10) -> list[dict]:
    """Confidence histogram in 0.1 buckets (FR lxviii)."""
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, max(0, int(float(v) * bins)))] += 1
    peak = max(counts + [1])
    return [{"label": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "count": n,
             "width": round(n / peak * 100, 1)} for i, n in enumerate(counts)]


def _review_errors(decided, critical: set[str]) -> dict:
    """False positives and negatives according to reviewers.

    False positive: the model named a critical class the reviewer rejected. False negative:
    the reviewer found a critical class the model missed.
    """
    per_class: dict[str, dict[str, int]] = {}
    fp = fn = agreed = 0
    for original, final, false_alarm in decided:
        final = final or original
        if original == final and not false_alarm:
            agreed += 1
        if original in critical and (final != original or false_alarm):
            fp += 1
            per_class.setdefault(original, {"fp": 0, "fn": 0})["fp"] += 1
        if final in critical and original != final:
            fn += 1
            per_class.setdefault(final, {"fp": 0, "fn": 0})["fn"] += 1
    return {"reviewed": len(decided), "confirmed": agreed, "false_positives": fp,
            "false_negatives": fn,
            "per_class": [{"class": k, **v} for k, v in sorted(per_class.items())]}


def _alert_response(rows) -> dict:
    """Alert acknowledge/resolve times and false-alarm count."""
    import statistics

    ack = [(a - c).total_seconds() for _, _, c, a, _, _ in rows if a and c]
    res = [(r - c).total_seconds() for _, _, c, _, r, _ in rows if r and c]
    by_status: dict[str, int] = {}
    for status, *_ in rows:
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "total": len(rows), "by_status": by_status,
        "false_alarms": sum(1 for *_, fa in rows if fa),
        "median_ack_sec": round(statistics.median(ack), 1) if ack else None,
        "median_resolve_sec": round(statistics.median(res), 1) if res else None,
        "unacknowledged": sum(1 for _, _, _, a, _, _ in rows if a is None),
    }




@admin_bp.get("/config")
@capability_required("edit_config")
def config():
    """FR liii / lxxx: configuration view. Edits go through PUT /api/admin/config/<file>."""
    store = _store()
    snapshot = store.snapshot()
    return _render(
        "admin/config.html",
        config_dir=str(store.config_dir),
        alert_rules_dir=str(store.alert_rules_dir),
        snapshot=snapshot,
        hashes=snapshot.to_dict().get("content_hashes") if hasattr(snapshot, "to_dict") else None,
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
    """FR i-ii: users and roles."""
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


@models_bp.get("/", endpoint="list")
@login_required
def model_list():
    """FR lxxv: model versions in service. Changing them needs manage_models."""
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




@visuals_bp.get("/events/<int:event_id>/visuals")
@owner_or_capability("download_any_audio")
def visuals(event_id: int):
    """Waveform peaks and spectrogram of the stored original recording (cached)."""
    from src.services.visuals import build_visuals

    with session_scope(_factory()) as session:
        event = _visible_event(session, event_id, for_download=True)
        audio = event.audio_file
        if audio is None:
            raise ApiError("not_found", "That event has no stored recording.")
        stored_path = audio.stored_path
        duration = audio.duration_sec
        sample_rate = audio.sample_rate

    try:
        target = current_app.config["SST_STORAGE"].resolve(stored_path)
    except ValueError:
        raise ApiError("storage_error", "That recording could not be located.")
    if not stored_path or not target.is_file():
        # The file may have been purged; the event is still there.
        raise ApiError(
            "no_audio",
            "That recording is no longer on disk. Its analysis and history are unaffected.",
        )

    payload = build_visuals(
        current_app.config["SST_STORAGE"],
        stored_path,
        event_id=event_id,
        fallback_duration=duration,
        fallback_sample_rate=sample_rate,
    )
    return jsonify(payload)
