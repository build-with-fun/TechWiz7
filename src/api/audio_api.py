"""Audio upload and download.

POST /api/audio/upload checks size and type before decoding, refuses an exact duplicate
(SHA-256) with 409 unless allow_duplicate is set, lets the pipeline analyse the clip, and
persists the result. Unusable audio is a 422 naming the reason; a near-duplicate is linked,
not refused.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from flask import Blueprint, current_app, request, send_file
from sqlalchemy import select

from src.auth import capability_required, client_ip, current_user
from src.db import session_scope
from src.errors import ApiError, current_request_id, validation_error
from src.models import AudioFile
from src.services.config import get_store

bp = Blueprint("audio_api", __name__, url_prefix="/api/audio")

_LOGGER = logging.getLogger(__name__)

# The contract's size gate applies to the whole request body, not to a decoded buffer.
_MAX_BODY_BYTES = 64 * 1024 * 1024

# The suffix check is cheap; the decoder has the final word. .webm is for live recordings.
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
    """File suffix from the declared mimetype, else the name. The stored file is named with it, so a
    wrong one would break playback.
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
    """Analyse an uploaded clip: 201 with the Event, 409 for a known duplicate, 422 for unusable
    audio.
    """
    from src.services.persistence import make_persistence_callback
    from src.services.pipeline import ModelsUnavailable, PipelineError, get_pipeline

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
    # FR lxxix: a microphone recording needs an explicit consent acknowledgement, stored on the file
    # row.
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
            "This file type is not supported. Send WAV, MP3, FLAC, OGG or M4A audio.",
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
            # FR lxxiii: name the first event so the operator can tell a re-upload from a real
            # repeat.
            raise ApiError(
                "duplicate_audio",
                f"These bytes are already stored as {existing.audio_id}. "
                "Re-send with allow_duplicate=true to store and analyse them again.",
                details={"audio_id": existing.audio_id, "sha256": digest},
            )

    try:
        pipeline = get_pipeline()
    except PipelineError as exc:
        # get_pipeline() raises the base PipelineError when nothing was loaded; catching only
        # ModelsUnavailable let it become a 500 instead of the documented 503.
        if isinstance(exc, ModelsUnavailable):
            # Name the missing artifact; a pathless 'not found' wastes the next person's time.
            raise ApiError("pipeline_unavailable", str(exc)) from exc
        # Never initialised: an internal wiring detail, so use the standard user-facing sentence.
        raise ApiError("pipeline_unavailable") from exc
    storage = current_app.config["SST_STORAGE"]
    actor = current_user._get_current_object() if hasattr(current_user, "_get_current_object") else current_user

    # The pipeline does the analysis; persistence is its callback, bound to this request's actor.
    persist = make_persistence_callback(
        storage=storage,
        actor=actor,
        request_id=current_request_id(),
        allow_duplicate=allow_duplicate,
    )
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        candidates = session.execute(
            select(AudioFile.audio_id, AudioFile.perceptual_fingerprint, AudioFile.stored_path)
            .where(AudioFile.perceptual_fingerprint.is_not(None))
            .order_by(AudioFile.created_at.desc())
            .limit(500)
        ).all()
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
                "near_duplicate_candidates": [
                    (audio_id, fingerprint, storage.resolve(stored) if stored else None)
                    for audio_id, fingerprint, stored in candidates],
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

    if (record.get("stored") or {}).get("error") or (
        record.get("status") == "analysed" and not record.get("event_id")
    ):
        _LOGGER.error("analysis could not be stored for request %s: %s",
                      current_request_id(), (record.get("stored") or {}).get("error"))
        raise ApiError("storage_error", "The analysis finished, but the result could not be saved. Try again.")
    # The pipeline record never embeds the actor, and the response builder only has the
    # ConfigStore, so attach it here where the request's user is still in scope. Without this
    # an event stored with created_by_id=2 reports created_by=null in the upload response.
    if record.get("created_by") is None:
        actor_obj = (current_user._get_current_object()
                     if hasattr(current_user, "_get_current_object") else current_user)
        if actor_obj is not None and getattr(actor_obj, "id", None):
            record["created_by"] = {
                "id": actor_obj.id, "username": actor_obj.username, "role": actor_obj.role,
            }
    return {"data": _response_for(record, store)}, 201


@bp.get("/<int:audio_pk>/download")
@capability_required("download_any_audio")
def download(audio_pk: int):
    """Download a stored file (reviewer and above). Paths resolve inside the storage root, so ../../
    cannot escape.
    """
    factory = current_app.config["SST_SESSION_FACTORY"]
    with session_scope(factory) as session:
        audio = session.execute(
            select(AudioFile).where(AudioFile.id == audio_pk)
        ).scalar_one_or_none()
        if audio is None:
            raise ApiError("not_found", "No audio file has that id.")
        if not audio.stored_path:
            raise ApiError("no_audio", "This recording has reached its retention limit.")
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


# Internals

def _request_body() -> bytes:
    """The uploaded bytes, size-checked before Flask's multipart parser can buffer a huge body."""
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
    if status == "error":
        _LOGGER.error("analysis failed at %s for request %s",
                      (record.get("error") or {}).get("stage"), current_request_id())
        raise ApiError("model_unavailable", "The models could not analyse this clip. Try again shortly.")
    if status == "rejected":
        # Unusable or undecodable: a 422 naming the reason, not a 201 with an empty event.
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
            },
        )
    return event_to_dict(record, store)
