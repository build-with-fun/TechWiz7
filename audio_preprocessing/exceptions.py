"""Explicit, explainable rejection of audio.

Owner: taha.  SRS Step 3 (file validation), FR viii, xii, xiii, lxxvii (error handling).

The module's one rule: **never silently drop unreadable audio.**  A file the pipeline
cannot use is rejected with a machine-readable reason code and a human-readable message,
because SRS Step 3 requires that "silent, damaged, unsupported, excessively short, or
unusable recordings must be rejected with appropriate messages", and a silently skipped
file in a batch upload is indistinguishable from a file that was never queued.

The reason codes are stable strings rather than enum members so they survive JSON
round-trips into the database, the audit trail and the API error payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------------------
# Reason codes
# --------------------------------------------------------------------------------------

UNREADABLE_FORMAT = "unreadable_format"
"""The container/codec could not be decoded at all (damaged or unknown)."""

CORRUPT_HEADER = "corrupt_header"
"""A recognisable container whose header or frames are damaged."""

UNSUPPORTED_FORMAT = "unsupported_format"
"""A decodable format outside the configured supported list (SRS FR iv)."""

EMPTY_AUDIO = "empty_audio"
"""The file is zero bytes or declares zero audio frames."""

TOO_LARGE = "too_large"
"""Exceeds the configured upload size limit (SRS FR viii)."""

TOO_LONG = "too_long"
"""Longer than the configured maximum analysis duration."""

TOO_SHORT = "too_short"
"""Shorter than the configured minimum usable duration (SRS Step 3)."""

SILENT = "silent"
"""Contains an audio stream but no usable signal (SRS FR xii)."""

NO_AUDIO_STREAM = "no_audio_stream"
"""The container has no audio stream (e.g. a video file with no audio track)."""

MISSING_FRAMES = "missing_frames"
"""The decoder reported unrecoverable gaps, so the waveform is not trustworthy."""

NO_SUCH_FILE = "no_such_file"
"""The path does not exist."""

REASON_DESCRIPTIONS: dict[str, str] = {
    UNREADABLE_FORMAT: "The file could not be decoded. It may be damaged or in a format we cannot read.",
    CORRUPT_HEADER: "The file header is damaged, so the audio cannot be trusted.",
    UNSUPPORTED_FORMAT: "This audio format is not supported. Supported formats: {supported}.",
    EMPTY_AUDIO: "The file is empty.",
    TOO_LARGE: "The file is larger than the {limit:.0f} MB upload limit.",
    TOO_LONG: "The recording is longer than the {limit:.0f} second analysis limit.",
    TOO_SHORT: "The recording is shorter than the {limit:.2f} second minimum.",
    SILENT: "The recording contains no audible signal.",
    NO_AUDIO_STREAM: "The file contains no audio track.",
    MISSING_FRAMES: "Parts of the audio could not be decoded, so the recording is incomplete.",
    NO_SUCH_FILE: "The file does not exist.",
}


@dataclass
class RejectionInfo:
    """Why a piece of audio was refused, in a form the UI and the audit trail can store."""

    reason: str
    detail: str = ""
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": self.reason, "detail": self.detail}
        if self.metrics:
            payload["metrics"] = self.metrics
        return payload


class AudioRejected(Exception):
    """Raised when audio genuinely cannot be used.

    Carries :class:`RejectionInfo` so the API layer can render the exact message and the
    test suite can assert on the reason code rather than on prose.
    """

    def __init__(self, reason: str, detail: str = "", metrics: dict[str, Any] | None = None):
        self.info = RejectionInfo(reason=reason, detail=detail, metrics=metrics)
        super().__init__(f"[{reason}] {detail}" if detail else f"[{reason}]")

    @property
    def reason(self) -> str:
        return self.info.reason

    def to_dict(self) -> dict[str, Any]:
        return self.info.to_dict()


class AudioDecodeError(AudioRejected):
    """Decoding failed outright."""

    def __init__(self, detail: str = "", metrics: dict[str, Any] | None = None):
        super().__init__(UNREADABLE_FORMAT, detail, metrics)


def message_for(reason: str, **fmt: Any) -> str:
    """Human-readable text for a reason code, with template values filled in.

    Kept as a function (not a dict lookup at the call site) so the wording lives in one
    place and an unknown code degrades to the code itself rather than to an empty string.
    """
    template = REASON_DESCRIPTIONS.get(reason)
    if template is None:
        return f"Audio rejected: {reason}"
    try:
        return template.format(**fmt)
    except (KeyError, IndexError, ValueError):
        return template
