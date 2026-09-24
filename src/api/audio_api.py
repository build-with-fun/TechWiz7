"""``/api/audio`` -- upload and retrieval of audio evidence.

Owner: sara.

The single busiest endpoint in the application: ``POST /api/audio/upload`` is where an
operator's clip becomes an event. The contract (``documentation/api_contract.md`` §3.3) fixes
the order of the steps, and the order matters as much as the steps:

1. size and type, before anything is decoded;
2. decode, and refuse what cannot be decoded or is too short;
3. sha256 of the bytes -- an exact duplicate is a ``409`` unless the caller asks for it;
4. near-duplicate by perceptual fingerprint, which is *not* a refusal, it is a link;
5. the pipeline decides, and its result is persisted by the ``persist`` callback.

Steps 3 and 4 are the reason this endpoint exists as more than a pass-through: ``409`` on a
duplicate and ``422`` on unusable audio are normal outcomes, and they are surfaced as
structured errors so the console renders them inline instead of as a red screen.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from sqlalchemy import select

from src.auth import capability_required, client_ip, current_user
from src.db import session_scope
from src.errors import ApiError, current_request_id, validation_error
from src.models import AudioFile, Event
from src.services.config import get_store

bp = Blueprint("audio_api", __name__, url_prefix="/api/audio")

_LOGGER = logging.getLogger(__name__)

# The contract's size gate applies to the whole request body, not to a decoded buffer.
_MAX_BODY_BYTES = 64 * 1024 * 1024

# Decoding is the expensive part, so the type check is done on the suffix first and the
# decoder has the final word anyway. '.webm' is here because the browser's MediaRecorder
# produces it for a live session, not because it is a preferred archive format.
_ACCEPTED_SUFFIXES = frozenset({".wav", ".wave", ".flac", ".mp3", ".ogg", ".oga", ".webm", ".m4a"})

_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".m4a": "audio/mp4",
}


def _sha256_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _suffix_of(filename: str | None, mimetype: str | None) -> str:
    """Prefer the declared mimetype, fall back to the name, and be honest about neither.

    A caller can send any mimetype it likes; the suffix is what the stored file is named
    after, so a lie here would write a .wav header on a .mp3 body and break playback later.
    """
    if mimetype:
        for suffix, mime in _MIME_BY_SUFFIX.items():
            if mimetype.lower() == mime:
                return suffix
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in _ACCEPTED_SUFFIXES:
            return suffix
    return ""


@bp.post("/upload")
@capability_required("upload_audio")
def upload():
    """Accept an audio clip and return the event the pipeline produced for it.

    ``201`` is the full ``Event`` (contract §3.1). A duplicate is ``409 duplicate_audio``
    unless ``?allow_duplicate=true`` is set, and unusable audio is ``422`` -- both are
    normal outcomes and both name the existing resource or the reason.
    """
    from src.services.persistence import make_persistence_callback
    from src.services.pipeline import get_pipeline

    store = get_store()
    raw = _request_body()
    filename = (request.form.get("filename") or request.files.get("file", None)
                and request.files["file"].filename) or "upload.wav"
    location = (request.form.get("location") or "").strip() or None
    source = (request.form.get("source") or "upload").strip().lower()
    allow_duplicate = (request.form.get("allow_duplicate", "") or
                       request.args.get("allow_duplicate", "")).strip().lower() in {"1", "true", "yes"}

    if source not in {"upload", "microphone"}:
        raise validation_error(
            "source must be one of: upload, microphone", source=source
        )
    # FR lxxix: a recording from the microphone needs an explicit acknowledgement that the
    # subject consented. The setting is stored on the file row, so it is auditable later.
    consent_ack = (request.form.get("consent_ack", "") or "").strip().lower() in {"1", "true", "yes"}
    if source == "microphone" and not consent_ack:
        raise validation_error(
            "consent_ack is required when source is microphone",
            consent_ack="true is required for a microphone recording",
        )

    suffix = _suffix_of(filename, request.files.get("file", None)
                        and request.files["file"].mimetype)
    if suffix not in _ACCEPTED_SUFFIXES:
        raise ApiError(
            "unsupported_media_type",
            "This file type is not supported. Send WAV, FLAC, MP3, OGG or WebM audio.",
            details={"filename": filename, "accepted": sorted(_ACCEPTED_SUFFIXES)},
        )

    digest = _sha256_of(raw)
    if not allow_duplicate:
        factory = current_app.config["SST_SESSION_FACTORY"]
        with session_scope(factory) as session:
            existing = session.execute(
                select(AudioFile).where(AudioFile.sha256 == digest)
            ).scalar_one_or_none()
        if existing is not None:
            # FR lxxiii: the bytes are already known. Name the first event so the operator
            # can decide whether this is a re-upload of the same evidence or a real repeat.
            raise ApiError(
                "duplicate_audio",
                f"These bytes are already stored as {existing.audio_id}. "
                "Re-send with allow_duplicate=true to store and analyse them again.",
                details={"audio_id": existing.audio_id, "sha256": digest},
            )

    pipeline = get_pipeline()
    storage = current_app.config["SST_STORAGE"]
    actor = current_user._get_current_object() if hasattr(current_user, "_get_current_object") else current_user

    # The pipeline owns steps 5-10 of the contract: preprocess, features, quality, both
    # models independently, comparison, severity, alert, review. Persistence is the
    # callback it calls at the end, bound to this request's actor and id so every row
    # written for this upload is attributable to it.
    persist = make_persistence_callback(
        storage=storage,
        actor=actor,
        request_id=current_request_id(),
    )
    try:
        record = pipeline.analyse_bytes(
            raw,
            filename=filename,
            origin="upload" if source == "upload" else "live",
            meta={
                "filename": filename,
                "location": location,
                "consent_acknowledged": consent_ack or None,
                "source": source,
                "client_ip": client_ip(),
                "content_type": request.files.get("file", None)
                and request.files["file"].mimetype,
            },
            persist=persist,
        )
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        _LOGGER.exception("analysis failed for %s", filename)
        raise ApiError(
            "internal_error",
            "The clip could not be analysed. The error has been logged with this request id.",
            details={"request_id": current_request_id(), "detail": type(exc).__name__},
        )

    return _response_for(record, store), 201


@bp.get("/<int:audio_pk>/download")
@capability_required("download_any_audio")
def download(audio_pk: int):
    """Stream the stored bytes of one audio file. Reviewer and above.

    The stored path is relative to the storage root and is resolved through the layout, so
    a path stored as ``../../etc/passwd`` can never be served.
    """
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        audio = session.execute(
            select(AudioFile).where(AudioFile.id == audio_pk)
        ).scalar_one_or_none()
        if audio is None:
            raise ApiError("not_found", "No audio file has that id.")
        path = current_app.config["SST_STORAGE"].resolve(audio.stored_path)
        if not path.is_file():
            _LOGGER.warning("stored audio is missing on disk: %s", audio.stored_path)
            raise ApiError(
                "storage_error",
                "The audio file is recorded but is no longer on disk.",
                details={"audio_id": audio.audio_id},
            )
        return send_file(
            str(path),
            mimetype=audio.content_type or "application/octet-stream",
            as_attachment=True,
            download_name=f"{audio.audio_id}{Path(audio.stored_path).suffix or '.wav'}",
        )


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------

def _request_body() -> bytes:
    """The uploaded bytes, after the size gate.

    Reads the part directly rather than through ``request.files`` first: the size gate has
    to fire before the whole body is buffered into memory, and Flask's multipart parser
    will happily buffer a 4 GB upload before handing it to the view.
    """
    if request.content_length is not None and request.content_length > _MAX_BODY_BYTES:
        raise ApiError(
            "payload_too_large",
            "The upload is larger than the 64 MB limit.",
            details={"limit_bytes": _MAX_BODY_BYTES, "received_bytes": request.content_length},
        )
    part = request.files.get("file")
    if part is None:
        raise validation_error("a 'file' part is required", file="required")
    raw = part.stream.read()
    if not raw:
        raise validation_error("the uploaded file is empty", file="cannot be empty")
    if len(raw) > _MAX_BODY_BYTES:
        raise ApiError(
            "payload_too_large",
            "The uploaded file is larger than the 64 MB limit.",
            details={"limit_bytes": _MAX_BODY_BYTES, "received_bytes": len(raw)},
        )
    return raw


def _response_for(record: dict, store) -> dict:
    """The contract's ``Event`` shape for a persisted analysis record."""
    from src.api.events_api import event_to_dict

    status = record.get("status")
    if status in {"rejected", "error"}:
        # FR xxxvii's unusable verdict and an undecodable clip both stop here. The caller
        # gets a 422 naming the reason, not a 201 with an empty event.
        rejection = record.get("rejection") or {}
        quality = record.get("quality") or {}
        code = "quality_unusable" if quality.get("verdict") == "Unusable" else "quality_rejected"
        raise ApiError(
            code,
            (rejection.get("message") or quality.get("summary")
             or "The clip did not meet the quality bar for analysis."),
            details={
                "verdict": quality.get("verdict"),
                "problems": quality.get("problems"),
                "audio_id": record.get("audio", {}).get("source_path"),
            },
        )
    return event_to_dict(record, store)
