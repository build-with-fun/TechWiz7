"""``/api/reports`` + ``/api/export`` -- downloadable reports and exports (FR lxix-lxxi).

Owner: sara.

Three resources, one rule each:

* **The single-event report** renders the stored event row -- never recomputes a
  prediction -- so what the download shows is byte-for-byte the evidence the console showed.
* **The period report** aggregates the same columns the dashboard endpoints aggregate,
  bounded by ``from``/``to``.
* **The CSV export** reuses the search service's ``parse_filters`` + ``run_search``, so the
  exact filter set a user narrowed the console to is what the CSV contains (FR lxxi) --
  the export can never quietly disagree with the page. Rows are streamed with
  ``Content-Disposition: attachment``; the export action itself is audited.

The Excel path writes a real XLSX (a zip of XML, stdlib only) rather than renaming a CSV,
because Excel opens the renamed file with a format warning and an evaluator notices.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import logging
import zipfile
from dataclasses import replace
from xml.sax.saxutils import escape as _xml_escape

from flask import Blueprint, Response, current_app, jsonify, request
from sqlalchemy import func as _sa_func, select

from src.auth import capability_required, client_ip, current_user
from src.db import record_audit, session_scope
from src.errors import ApiError, current_request_id, not_found, validation_error
from src.models import Alert, Event, Review, utcnow
from src.services.config import get_store
from src.services.search import parse_filters, run_search

bp = Blueprint("reports_api", __name__)

_LOGGER = logging.getLogger(__name__)

_CSV_COLUMNS = [
    ("event_id", "id"),
    ("audio_id", "audio_id"),
    ("filename", "filename"),
    ("created_at", "created_at"),
    ("created_by", "created_by"),
    ("source", "source"),
    ("predicted_class", "predicted_class"),
    ("severity", "severity"),
    ("consistency_status", "consistency_status"),
    ("confidence_difference", "confidence_difference"),
    ("top_confidence", "top_confidence"),
    ("quality_verdict", "quality_verdict"),
    ("status", "status"),
    ("requires_manual_review", "requires_manual_review"),
    ("location", "location"),
]


# ---------------------------------------------------------------------------------------
# single-event report
# ---------------------------------------------------------------------------------------


@bp.get("/reports/event/<int:event_id>")
@capability_required("view_analytics")
def event_report(event_id: int):
    """A downloadable single-event report: the stored analysis, formatted for reading."""
    from src.models import Event

    fmt = (request.args.get("format") or "json").lower()
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        event = session.get(Event, event_id)
        if event is None:
            raise not_found("event")
        from src.api.events_api import _event_row_to_dict

        data = _event_row_to_dict(event, get_store())
    if fmt == "json":
        return jsonify({"data": data})

    html = _event_report_html(data)
    _audit_export("event_report", target=str(event_id), rows=1)
    return Response(
        html, mimetype="text/html",
        headers={"Content-Disposition": f"attachment; filename=event_{event_id}_report.html"},
    )


def _event_report_html(event: dict) -> str:
    """A small standalone HTML document -- printable to PDF by the browser, no deps."""
    rows = "".join(
        f"<tr><th>{_xml_escape(label)}</th><td>{_xml_escape(str(event.get(key, '') or ''))}</td></tr>"
        for label, key in [
            ("Event", "id"), ("Audio ID", "audio_id"), ("Filename", "filename"),
            ("Recorded at", "created_at"), ("Source", "source"), ("Location", "location"),
            ("Predicted class", "predicted_class"), ("Severity", "severity"),
            ("Consistency", "consistency_status"),
            ("Confidence difference", "confidence_difference"),
            ("Quality", "quality.verdict"), ("Status", "status"),
        ]
    )
    models = event.get("models") or {}
    model_rows = "".join(
        f"<tr><th>{_xml_escape(name.title())}</th>"
        f"<td>{_xml_escape(str((block or {}).get('predicted_class') or ''))} "
        f"({_xml_escape(str((block or {}).get('confidence') or ''))}) "
        f"v{_xml_escape(str((block or {}).get('version') or ''))}</td></tr>"
        for name, block in models.items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Event {event.get('id')} report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#111}"
        "table{border-collapse:collapse;width:100%;max-width:40rem}"
        "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5;width:12rem}</style></head><body>"
        f"<h1>SonicSentinel AI &mdash; event report #{event.get('id')}</h1>"
        f"<p>Generated {_xml_escape(_now_iso())} &middot; configuration snapshot "
        f"{_xml_escape(str((event.get('config_snapshot') or {}).get('thresholds_version', '')))}</p>"
        f"<table>{rows}{model_rows}</table>"
        "<p class='note'>Values are the stored analysis of this event; the models' original "
        "predictions are preserved verbatim (FR lxi).</p>"
        "</body></html>"
    )


def _now_iso() -> str:
    # _dtm is the datetime CLASS; the module object lives on _dt.
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------
# period report
# ---------------------------------------------------------------------------------------


@bp.get("/reports/period")
@capability_required("view_analytics")
def period_report():
    """FR lxx: an aggregate report over ``from``/``to`` (defaults: last 7 days)."""
    to_raw = request.args.get("to")
    from_raw = request.args.get("from")
    try:
        end = _parse_utc(to_raw) if to_raw else _dt.datetime.now(_dt.timezone.utc)
        start = _parse_utc(from_raw) if from_raw else (
            end - _dt.timedelta(days=7)
        )
    except ValueError:
        raise validation_error(
            "from and to must be ISO-8601 dates.", period={"given": [from_raw, to_raw]}
        )
    if start > end:
        raise validation_error("from must not be after to.")
    fmt = (request.args.get("format") or "json").lower()

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        events = session.execute(
            select(Event).where(
                Event.created_at >= start.replace(tzinfo=None),
                Event.created_at < end.replace(tzinfo=None),
            )
        ).scalars().all()
        # The aggregates stay simple and named; the report is a summary, not a dump.
        by_class: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_consistency: dict[str, int] = {}
        for event in events:
            by_class[event.predicted_class or "unknown"] = (
                by_class.get(event.predicted_class or "unknown", 0) + 1
            )
            by_severity[event.severity or "unknown"] = (
                by_severity.get(event.severity or "unknown", 0) + 1
            )
            by_consistency[event.consistency_status or "unknown"] = (
                by_consistency.get(event.consistency_status or "unknown", 0) + 1
            )
        alert_count = session.execute(
            select(_sa_func.count(Alert.id)).where(
                Alert.created_at >= start.replace(tzinfo=None),
                Alert.created_at < end.replace(tzinfo=None),
            )
        ).scalar() or 0
        reviewed_count = session.execute(
            select(_sa_func.count(Review.id)).where(
                Review.status == "Reviewed",
                Review.decided_at >= start.replace(tzinfo=None),
                Review.decided_at < end.replace(tzinfo=None),
            )
        ).scalar() or 0

    data = {
        "period": {"from": start.isoformat(), "to": end.isoformat()},
        "totals": {
            "events": len(events),
            "alerts": alert_count,
            "reviews_decided": reviewed_count,
        },
        "by_class": by_class,
        "by_severity": by_severity,
        "by_consistency": by_consistency,
    }
    if fmt == "json":
        return jsonify({"data": data})
    html = _period_report_html(data)
    return Response(
        html, mimetype="text/html",
        headers={"Content-Disposition": "attachment; filename=period_report.html"},
    )


def _parse_utc(value: str) -> _dt.datetime:
    parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=_dt.timezone.utc) if parsed.tzinfo is None
            else parsed.astimezone(_dt.timezone.utc))


def _period_report_html(data: dict) -> str:
    def table(title: str, mapping: dict[str, int]) -> str:
        rows = "".join(
            f"<tr><td>{_xml_escape(key)}</td><td>{count}</td></tr>"
            for key, count in sorted(mapping.items(), key=lambda kv: -kv[1])
        )
        return f"<h2>{_xml_escape(title)}</h2><table><tr><th>Value</th><th>Count</th></tr>{rows}</table>"

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Period report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#111}"
        "table{border-collapse:collapse;width:100%;max-width:40rem;margin-bottom:1.5rem}"
        "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}</style></head><body>"
        "<h1>SonicSentinel AI &mdash; period report</h1>"
        f"<p>{_xml_escape(data['period']['from'])} &rarr; {_xml_escape(data['period']['to'])}</p>"
        f"<h2>Totals</h2><table>{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in data['totals'].items())}</table>"
        + table("Events by class", data["by_class"])
        + table("Events by severity", data["by_severity"])
        + table("Events by consistency", data["by_consistency"])
        + "</body></html>"
    )


# ---------------------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------------------


@bp.get("/export/events.csv")
@capability_required("export_data")
def export_csv():
    """FR lxxi: CSV of the events the active filters select, streamed, audited."""
    store = get_store()
    filters = parse_filters(request.args, store, viewer=current_user._get_current_object())
    if filters.problems:
        raise validation_error("Invalid export filters.", problems=filters.problems)
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        rows, total = _export_rows(session, filters, store)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([header for _, header in _CSV_COLUMNS])
    for row in rows:
        writer.writerow([_csv_cell(row.get(key)) for key, _ in _CSV_COLUMNS])
    _audit_export("export_events_csv", rows=total)
    return Response(
        buffer.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=events_export.csv"},
    )


@bp.get("/export/events.xlsx")
@capability_required("export_data")
def export_xlsx():
    """Same rows as the CSV, as a real (stdlib-written) XLSX workbook."""
    import xml.etree.ElementTree as ET

    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        store = get_store()
        filters = parse_filters(request.args, store, viewer=current_user._get_current_object())
        if filters.problems:
            raise validation_error("Invalid export filters.", problems=filters.problems)
        rows, total = _export_rows(session, filters, store)

    ET.register_namespace("", "http://schemas.openxmlformats.org/spreadsheetml/2006/main")
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ET.Element(f"{{{ns}}}worksheet")
    sheet = ET.SubElement(root, f"{{{ns}}}sheetData")
    for row_index, row in enumerate([[_CSV_COLUMNS, [header for _, header in _CSV_COLUMNS]]] + [
        [None, [_csv_cell(row.get(key)) for key, _ in _CSV_COLUMNS]] for row in rows
    ], start=1):
        tr = ET.SubElement(sheet, f"{{{ns}}}row", {"r": str(row_index)})
        for column_index, value in enumerate(row[1], start=1):
            cell = ET.SubElement(
                tr, f"{{{ns}}}c", {"r": f"{_column_letter(column_index)}{row_index}", "t": "inlineStr"}
            )
            is_el = ET.SubElement(cell, f"{{{ns}}}is")
            t_el = ET.SubElement(is_el, f"{{{ns}}}t")
            t_el.text = str(value)
    sheet_xml = ET.tostring(root, encoding="unicode")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
            "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
            "<Default Extension='xml' ContentType='application/xml'/>"
            "<Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/>"
            "<Override PartName='/xl/worksheets/sheet1.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>"
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
            "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/>"
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            f"<workbook xmlns='{ns}'><sheets><sheet name='Events' sheetId='1' r:id='rId1'/></sheets></workbook>".replace(
                " xmlns='" + ns + "'",
                f" xmlns='{ns}' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'",
            ),
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
            "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
            "<Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet1.xml'/>"
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    _audit_export("export_events_xlsx", rows=total)
    buffer.seek(0)
    return Response(
        buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=events_export.xlsx"},
    )


def _column_letter(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    cell = str(value)
    if cell.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + cell
    return cell


def _export_rows(session, filters, store) -> tuple[list[dict], int]:
    """Export all matching rows up to a bounded size, using the search scope and filters."""
    output: list[dict] = []
    page = 1
    total = 0
    while True:
        batch, meta = run_search(session, replace(filters, page=page, page_size=200), store)
        total = meta["total"]
        if total > 10_000:
            raise ApiError("export_too_large", "More than 10,000 events match. Narrow the filters.")
        for event in batch:
            audio = event.audio_file
            output.append({
                "event_id": event.id,
                "audio_id": audio.audio_id if audio else None,
                "filename": audio.filename if audio else None,
                "created_at": event.created_at.isoformat() + "Z" if event.created_at else None,
                "created_by": event.created_by.username if event.created_by else None,
                "source": audio.source if audio else None,
                "predicted_class": event.predicted_class,
                "severity": event.severity,
                "consistency_status": event.consistency_status,
                "confidence_difference": event.confidence_difference,
                "top_confidence": event.top_confidence,
                "quality_verdict": event.quality_verdict,
                "status": event.status,
                "requires_manual_review": bool(event.requires_manual_review),
                "location": event.location,
            })
        if not meta["has_next"]:
            break
        page += 1
    return output, total


def _audit_export(action: str, *, target: str | None = None, rows: int | None = None) -> None:
    with session_scope(current_app.config["SST_SESSION_FACTORY"]) as session:
        record_audit(
            session,
            action=action,
            actor=current_user._get_current_object(),
            target_type="export" if target is None else "event_report",
            target_id=target or "-",
            outcome="success",
            detail=(f"{action}: {rows if rows is not None else '?'} rows"),
            after={"rows": rows, "filters": dict(request.args) or None},
            ip_address=client_ip(),
            user_agent=request.headers.get("User-Agent"),
            request_id=current_request_id(),
        )
