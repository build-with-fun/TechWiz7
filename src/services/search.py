"""The event search.  SRS FR lxvii, and the scope rule from FR lxiii/lxvii.

Owner: sara.

The filter vocabulary lives in one place because two consumers share it and must not drift:

* the HTML search page, which renders its form from :func:`filter_fields`, and
* the JSON endpoints, whose query parameters are the same names.

Whoever parses ``request.args`` should call :func:`parse_filters`, which both validates and
*names* every problem, and :func:`build_query`, which applies the scope rule:

    a normal user sees only their own events; the fleet-wide roles see everything, and may
    narrow to one user.

The scope rule is applied in the query, never by filtering a page of results afterwards --
otherwise the counts and the pagination would describe a set the user cannot see.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session, selectinload

from src.errors import ApiError
from src.models import (
    CONSISTENCY_STATUSES,
    EVENT_STATUSES,
    QUALITY_VERDICTS,
    AudioFile,
    Event,
    Review,
    User,
    utcnow,
)

#: The order options appear in the UI, with the label the page shows. Kept beside the field
#: definitions so a new sort order is one entry, not three edits.
SORT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("newest", "Newest first"),
    ("oldest", "Oldest first"),
    ("confidence_desc", "Highest confidence"),
    ("confidence_asc", "Lowest confidence"),
    ("difference_desc", "Largest model disagreement"),
    ("severity_desc", "Most severe first"),
    ("class_asc", "Sound category (A-Z)"),
)

#: Severity order for sorting, taken from the configured scale at query time -- see
#: ``build_query``. This fallback only matters if a caller asks for a bad sort key.
_SEVERITY_ORDER_FALLBACK = ("Informational", "Low", "Medium", "High", "Critical")


@dataclass
class Filters:
    """A validated search. Every field is either absent or a real value."""

    audio_id: str | None = None
    filename: str | None = None
    sound_class: str | None = None
    severity: tuple[str, ...] = ()
    quality: tuple[str, ...] = ()
    review_status: str | None = None
    status: tuple[str, ...] = ()
    consistency_status: tuple[str, ...] = ()
    user: str | None = None
    source: str | None = None
    date_from: dt.datetime | None = None
    date_to: dt.datetime | None = None
    confidence_min: float | None = None
    confidence_max: float | None = None
    difference_min: float | None = None
    difference_max: float | None = None
    needs_review: bool | None = None
    critical_only: bool | None = None
    q: str | None = None
    sort: str = "newest"
    page: int = 1
    page_size: int = 25
    #: The scope the caller is allowed to see, applied as a hard constraint.
    viewer_id: int | None = None
    viewer_sees_all: bool = False
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe form, for the API's ``meta.filters`` echo."""
        return {
            "audio_id": self.audio_id,
            "filename": self.filename,
            "sound_class": self.sound_class,
            "severity": list(self.severity),
            "quality": list(self.quality),
            "review_status": self.review_status,
            "status": list(self.status),
            "consistency_status": list(self.consistency_status),
            "user": self.user,
            "source": self.source,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "difference_min": self.difference_min,
            "difference_max": self.difference_max,
            "needs_review": self.needs_review,
            "critical_only": self.critical_only,
            "q": self.q,
            "sort": self.sort,
            "page": self.page,
            "page_size": self.page_size,
        }

    @property
    def is_filtered(self) -> bool:
        """True when the user narrowed something -- the page then offers a "clear" link."""
        return bool(
            self.audio_id or self.filename or self.sound_class or self.severity
            or self.quality or self.review_status or self.status or self.consistency_status
            or self.user or self.source or self.date_from or self.date_to
            or self.confidence_min is not None or self.confidence_max is not None
            or self.difference_min is not None or self.difference_max is not None
            or self.needs_review is not None or self.critical_only is not None or self.q
        )


def filter_fields(store) -> list[dict[str, Any]]:
    """The filter form, as data. The template renders this; nobody hardcodes a field.

    Every option list comes from the configuration files, so when an evaluator adds a class
    to ``config/classes.json`` the dropdown grows without a template edit.
    """
    return [
        {"name": "audio_id", "label": "Audio ID", "type": "text",
         "placeholder": "SST-2026-09-23-000007", "srs": "FR lxvii"},
        {"name": "filename", "label": "Filename", "type": "text",
         "placeholder": "clip_012.wav", "srs": "FR lxvii"},
        {"name": "sound_class", "label": "Sound category", "type": "select",
         "options": [{"value": c, "label": c} for c in store.class_names()],
         "selected": None, "srs": "FR lxvii"},
        {"name": "severity", "label": "Severity", "type": "multi",
         "options": [{"value": s, "label": s} for s in store.severity_scale()],
         "srs": "FR lxvii"},
        {"name": "quality", "label": "Audio quality", "type": "multi",
         "options": [{"value": q, "label": q} for q in store.quality_ordering()],
         "srs": "FR lxvii"},
        {"name": "status", "label": "Event status", "type": "multi",
         "options": [{"value": s, "label": s} for s in EVENT_STATUSES],
         "srs": "FR lxii"},
        {"name": "review_status", "label": "Review status", "type": "select",
         "options": [
             {"value": "queued", "label": "Awaiting review"},
             {"value": "decided", "label": "Reviewed"},
             {"value": "not_queued", "label": "Never queued"},
         ], "srs": "FR lxvii"},
        {"name": "consistency_status", "label": "Model consistency", "type": "multi",
         "options": [{"value": c, "label": c} for c in CONSISTENCY_STATUSES],
         "srs": "FR xxxiii"},
        {"name": "user", "label": "User", "type": "text",
         "placeholder": "username (fleet-wide roles only)",
         "requires_capability": "search_all_events", "srs": "FR lxvii"},
        {"name": "date_from", "label": "From", "type": "date", "srs": "FR lxvii"},
        {"name": "date_to", "label": "To", "type": "date", "srs": "FR lxvii"},
        {"name": "confidence_min", "label": "Confidence at least", "type": "number",
         "min": 0, "max": 1, "step": 0.01, "srs": "FR lxvii"},
        {"name": "confidence_max", "label": "Confidence at most", "type": "number",
         "min": 0, "max": 1, "step": 0.01, "srs": "FR lxvii"},
        {"name": "needs_review", "label": "Only items needing review", "type": "checkbox",
         "srs": "FR lvii"},
        {"name": "critical_only", "label": "Only critical categories", "type": "checkbox",
         "srs": "FR liii"},
        {"name": "q", "label": "Free text (ID, filename, class, location)", "type": "text",
         "srs": "FR lxvii"},
        {"name": "sort", "label": "Sort by", "type": "select",
         "options": [{"value": v, "label": l} for v, l in SORT_OPTIONS],
         "selected": "newest", "srs": "FR lxvii"},
    ]


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------

_MULTI_FIELDS = {"severity", "quality", "status", "consistency_status"}


def _as_float(value: str | None, field_name: str, problems: list[str]) -> float | None:
    """A confidence value. Out of [0, 1] is a *reported* problem, not a silent clamp:
    silently ignoring `confidence_min=5` would show unfiltered results to someone who
    believes they filtered."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        problems.append(f"{field_name} must be a number between 0 and 1")
        return None
    if not 0.0 <= number <= 1.0:
        problems.append(f"{field_name} must be between 0 and 1")
        return None
    return number


def _as_date(value: str | None, field_name: str, problems: list[str],
             *, end_of_day: bool = False) -> dt.datetime | None:
    """Accept a plain date or a full timestamp. A plain end date covers that whole day,
    because "to 23 September" meaning "up to midnight at its start" drops a day of results."""
    if value in (None, ""):
        return None
    for parse in (
        lambda v: dt.datetime.fromisoformat(v),
        lambda v: dt.datetime.strptime(v, "%Y-%m-%d"),
    ):
        try:
            parsed = parse(value)
        except (TypeError, ValueError):
            continue
        if end_of_day and len(value.strip()) <= 10:
            parsed = parsed + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
        return parsed
    problems.append(f"{field_name} must be a date like 2026-09-23")
    return None


def _has(args: Mapping[str, Any], key: str) -> bool:
    value = args.get(key)
    return value not in (None, "")


def _multi(args: Mapping[str, Any], key: str, allowed: Iterable[str],
           problems: list[str]) -> tuple[str, ...]:
    """Read a repeatable parameter (``?severity=High&severity=Critical``) or a comma list.

    Both spellings are accepted because the HTML form sends repeated values and a JSON client
    naturally sends one comma-separated string.
    """
    raw: list[str] = []
    if hasattr(args, "getlist"):
        raw = [str(v) for v in args.getlist(key)]  # type: ignore[attr-defined]
    elif _has(args, key):
        raw = [str(args[key])]
    values: list[str] = []
    for item in raw:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    allowed_set = {a.lower(): a for a in allowed}
    resolved: list[str] = []
    for value in values:
        match = allowed_set.get(value.lower())
        if match is None:
            problems.append(f"{key} has an unknown value: {value!r}")
        elif match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def parse_filters(args: Mapping[str, Any], store, *, viewer=None,
                  default_page_size: int = 25, max_page_size: int = 200) -> Filters:
    """Turn query parameters into a validated :class:`Filters`.

    Never raises on a bad value: it collects the problem and carries on, so the search page
    can show "confidence_min must be between 0 and 1" next to the form while still rendering
    the results it *can* produce. A bad *page* is the exception -- see below.
    """
    problems: list[str] = []
    filters = Filters(problems=problems)

    filters.audio_id = (args.get("audio_id") or "").strip() or None
    filters.filename = (args.get("filename") or "").strip() or None
    filters.sound_class = (args.get("sound_class") or "").strip() or None
    filters.q = (args.get("q") or "").strip() or None
    filters.source = (args.get("source") or "").strip() or None
    filters.user = (args.get("user") or "").strip() or None

    if filters.sound_class and filters.sound_class not in store.class_names():
        problems.append(f"sound_class has an unknown value: {filters.sound_class!r}")
        filters.sound_class = None

    filters.severity = _multi(args, "severity", store.severity_scale(), problems)
    filters.quality = _multi(args, "quality", store.quality_ordering(), problems)
    filters.status = _multi(args, "status", EVENT_STATUSES, problems)
    filters.consistency_status = _multi(args, "consistency_status", CONSISTENCY_STATUSES,
                                        problems)

    review_status = (args.get("review_status") or "").strip().lower() or None
    if review_status and review_status not in {"queued", "decided", "not_queued", "any"}:
        problems.append(f"review_status has an unknown value: {review_status!r}")
        review_status = None
    filters.review_status = None if review_status in (None, "any") else review_status

    filters.date_from = _as_date(args.get("date_from"), "date_from", problems)
    filters.date_to = _as_date(args.get("date_to"), "date_to", problems, end_of_day=True)
    if filters.date_from and filters.date_to and filters.date_from > filters.date_to:
        problems.append("date_from is after date_to, so no event can match")

    filters.confidence_min = _as_float(args.get("confidence_min"), "confidence_min", problems)
    filters.confidence_max = _as_float(args.get("confidence_max"), "confidence_max", problems)
    filters.difference_min = _as_float(args.get("difference_min"), "difference_min", problems)
    filters.difference_max = _as_float(args.get("difference_max"), "difference_max", problems)
    if (filters.confidence_min is not None and filters.confidence_max is not None
            and filters.confidence_min > filters.confidence_max):
        problems.append("confidence_min is greater than confidence_max, so nothing can match")

    if _has(args, "needs_review"):
        filters.needs_review = _truthy(args["needs_review"])
    if _has(args, "critical_only"):
        filters.critical_only = _truthy(args["critical_only"])

    sort = (args.get("sort") or "newest").strip().lower()
    if sort not in {v for v, _ in SORT_OPTIONS}:
        problems.append(f"sort has an unknown value: {sort!r}")
        sort = "newest"
    filters.sort = sort

    # Pagination. A bad page is fatal rather than forgiving: unlike a filter, it decides
    # *which rows are returned*, and quietly returning page 1 when page 99 was asked for
    # would make a client believe it had seen everything.
    page_raw = args.get("page")
    if _has(args, "page"):
        try:
            filters.page = int(page_raw)
        except (TypeError, ValueError):
            raise ApiError("validation_error", "page must be a whole number.",
                           details={"page": "must be an integer"})
        if filters.page < 1:
            raise ApiError("validation_error", "page must be 1 or greater.",
                           details={"page": "must be >= 1"})
    size_raw = args.get("page_size")
    if _has(args, "page_size"):
        try:
            filters.page_size = int(size_raw)
        except (TypeError, ValueError):
            raise ApiError("validation_error", "page_size must be a whole number.",
                           details={"page_size": "must be an integer"})
        if filters.page_size < 1:
            raise ApiError("validation_error", "page_size must be 1 or greater.",
                           details={"page_size": "must be >= 1"})
        if filters.page_size > max_page_size:
            # Clamped, not refused: a client asking for too much gets the maximum, which is
            # what it wanted anyway, and the response's meta says what was actually used.
            filters.page_size = max_page_size
    else:
        filters.page_size = default_page_size

    # Scope. This is the authorisation decision for *reading*, made once, here.
    if viewer is not None:
        filters.viewer_id = getattr(viewer, "id", None)
        filters.viewer_sees_all = bool(
            hasattr(viewer, "can") and viewer.can("search_all_events")
        )
        if filters.user and not filters.viewer_sees_all:
            # Asking for another user's events is not a filter the viewer may apply. Drop it
            # and say so, rather than returning nothing for a reason the user cannot see.
            problems.append(
                "the 'user' filter needs the fleet-wide search permission; it was ignored"
            )
            filters.user = None
    return filters


# ---------------------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------------------


def build_query(filters: Filters, store):
    """The ``select`` for this search, with the scope rule applied as a WHERE clause."""
    statement = select(Event).join(AudioFile, Event.audio_file_id == AudioFile.id)

    conditions: list[Any] = []

    if not filters.viewer_sees_all:
        if filters.viewer_id is None:
            # A viewer who is neither signed in nor fleet-wide must match nothing. Failing
            # closed here means a future caller that forgets to pass a viewer gets an empty
            # result rather than everyone's events.
            return statement.where(Event.id < 0)
        conditions.append(Event.created_by_id == filters.viewer_id)

    if filters.audio_id:
        conditions.append(AudioFile.audio_id.ilike(f"%{filters.audio_id}%"))
    if filters.filename:
        conditions.append(AudioFile.filename.ilike(f"%{filters.filename}%"))
    if filters.sound_class:
        conditions.append(Event.predicted_class == filters.sound_class)
    if filters.severity:
        conditions.append(Event.severity.in_(filters.severity))
    if filters.quality:
        conditions.append(Event.quality_verdict.in_(filters.quality))
    if filters.status:
        conditions.append(Event.status.in_(filters.status))
    if filters.consistency_status:
        conditions.append(Event.consistency_status.in_(filters.consistency_status))
    if filters.source:
        conditions.append(Event.source == filters.source)
    if filters.date_from:
        conditions.append(Event.created_at >= filters.date_from)
    if filters.date_to:
        conditions.append(Event.created_at <= filters.date_to)
    if filters.confidence_min is not None:
        conditions.append(Event.top_confidence >= filters.confidence_min)
    if filters.confidence_max is not None:
        conditions.append(Event.top_confidence <= filters.confidence_max)
    if filters.difference_min is not None:
        conditions.append(Event.confidence_difference >= filters.difference_min)
    if filters.difference_max is not None:
        conditions.append(Event.confidence_difference <= filters.difference_max)
    if filters.needs_review is True:
        conditions.append(Event.requires_manual_review.is_(True))
    elif filters.needs_review is False:
        conditions.append(Event.requires_manual_review.is_(False))
    if filters.critical_only is True:
        conditions.append(Event.severity == "Critical")
    elif filters.critical_only is False:
        conditions.append(Event.severity != "Critical")

    if filters.user:
        # Either the uploader or the reviewer who touched it -- "show me everything omar
        # was involved in" is the question an investigator actually asks.
        conditions.append(
            or_(
                Event.created_by_id.in_(
                    select(User.id).where(User.username.ilike(f"%{filters.user}%"))
                ),
                Event.id.in_(
                    select(Review.event_id).where(
                        Review.decided_by_id.in_(
                            select(User.id).where(User.username.ilike(f"%{filters.user}%"))
                        )
                    )
                ),
            )
        )

    if filters.review_status == "queued":
        conditions.append(Event.requires_manual_review.is_(True))
        conditions.append(Event.status.in_(("Manual Review", "Uncertain")))
    elif filters.review_status == "decided":
        conditions.append(Event.status == "Reviewed")
    elif filters.review_status == "not_queued":
        conditions.append(Event.requires_manual_review.is_(False))

    if filters.q:
        needle = f"%{filters.q}%"
        conditions.append(
            or_(
                AudioFile.audio_id.ilike(needle),
                AudioFile.filename.ilike(needle),
                Event.predicted_class.ilike(needle),
                Event.final_class.ilike(needle),
                Event.location.ilike(needle),
                Event.review_reason.ilike(needle),
            )
        )

    if conditions:
        statement = statement.where(and_(*conditions))

    return statement.order_by(*_order_by(filters.sort, store))


def _order_by(sort: str, store) -> Sequence[Any]:
    """Sort keys, each with a deterministic tiebreak on ``id``.

    Without the tiebreak, two events created in the same second can swap places between
    pages and a user paging through the results sees one twice and misses another.
    """
    tiebreak = Event.id.desc()
    if sort == "newest":
        return (Event.created_at.desc(), tiebreak)
    if sort == "oldest":
        return (Event.created_at.asc(), Event.id.asc())
    if sort == "confidence_desc":
        return (Event.top_confidence.desc().nullslast(), tiebreak)
    if sort == "confidence_asc":
        return (Event.top_confidence.asc().nullsfirst(), Event.id.asc())
    if sort == "difference_desc":
        return (Event.confidence_difference.desc().nullslast(), tiebreak)
    if sort == "severity_desc":
        # Ranked by the configured severity scale, so adding a level to
        # alert_rules/severity_levels.json changes the sort without a code change.
        # A CASE is used rather than alphabetical order because "Critical" must outrank
        # "Low" -- alphabetically it would not.
        scale = list(store.severity_scale()) or list(_SEVERITY_ORDER_FALLBACK)
        ranking = case(
            {name: index for index, name in enumerate(scale)},
            value=Event.severity,
            else_=-1,
        )
        return (ranking.desc(), tiebreak)
    if sort == "class_asc":
        return (Event.predicted_class.asc(), Event.id.asc())
    return (Event.created_at.desc(), tiebreak)


def run_search(session: Session, filters: Filters, store) -> tuple[list[Event], dict[str, Any]]:
    """Run the search and return the page of events plus the metadata the page needs.

    ``meta`` carries the totals, the page bounds and a "what you are looking at" sentence,
    so the page never has to recompute a count and never shows a number that disagrees with
    the rows beneath it.
    """
    statement = build_query(filters, store)

    count_statement = select(func.count()).select_from(statement.order_by(None).subquery())
    total = session.execute(count_statement).scalar() or 0

    rows = session.execute(
        statement
        .options(selectinload(Event.audio_file), selectinload(Event.confidence_scores),
                 selectinload(Event.alerts), selectinload(Event.created_by))
        .offset((filters.page - 1) * filters.page_size)
        .limit(filters.page_size)
    ).scalars().all()

    page_count = max(1, (total + filters.page_size - 1) // filters.page_size)
    meta = {
        "total": total,
        "page": filters.page,
        "page_size": filters.page_size,
        "page_count": page_count,
        "has_next": filters.page < page_count,
        "has_previous": filters.page > 1,
        "shown": len(rows),
        "first_row": (filters.page - 1) * filters.page_size + 1 if total else 0,
        "last_row": (filters.page - 1) * filters.page_size + len(rows),
        "scope": "all users" if filters.viewer_sees_all else "your own uploads",
        "filters": filters.to_dict(),
        "filtered": filters.is_filtered,
        "sort_label": dict(SORT_OPTIONS).get(filters.sort, filters.sort),
    }
    return list(rows), meta


def describe_filters(filters: Filters) -> str:
    """A plain sentence naming what is being filtered, shown above the results.

    It matters more than it looks: a user who is puzzled by an empty list needs to see that
    a filter they set an hour ago is still applied.
    """
    parts: list[str] = []
    if filters.audio_id:
        parts.append(f"audio ID contains {filters.audio_id!r}")
    if filters.filename:
        parts.append(f"filename contains {filters.filename!r}")
    if filters.sound_class:
        parts.append(f"category is {filters.sound_class}")
    if filters.severity:
        parts.append("severity is " + " or ".join(filters.severity))
    if filters.quality:
        parts.append("quality is " + " or ".join(filters.quality))
    if filters.status:
        parts.append("status is " + " or ".join(filters.status))
    if filters.consistency_status:
        parts.append("model consistency is " + " or ".join(filters.consistency_status))
    if filters.date_from:
        parts.append(f"created on or after {filters.date_from:%Y-%m-%d}")
    if filters.date_to:
        parts.append(f"created on or before {filters.date_to:%Y-%m-%d}")
    if filters.confidence_min is not None:
        parts.append(f"confidence at least {filters.confidence_min:.2f}")
    if filters.confidence_max is not None:
        parts.append(f"confidence at most {filters.confidence_max:.2f}")
    if filters.needs_review is True:
        parts.append("awaiting review")
    if filters.critical_only is True:
        parts.append("critical categories only")
    if filters.user:
        parts.append(f"user matches {filters.user!r}")
    if filters.q:
        parts.append(f"matching {filters.q!r}")
    if not parts:
        return "Everything, most recent first."
    return "Filtered by " + ", ".join(parts) + "."
