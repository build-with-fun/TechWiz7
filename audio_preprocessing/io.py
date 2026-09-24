"""Reading audio in, and the file-level validation SRS Step 3 demands.

Owner: taha.  SRS Step 3, Step 4 (format conversion), FR iv, viii, ix, x, lxxvii.

Decoding strategy, in order:

1. ``soundfile`` -- fast, sample-exact, and the normal path for WAV/FLAC/OGG/MP3.
2. ``ffmpeg`` -- the fallback for M4A/AAC and for anything soundfile declines, decoded to
   a temporary WAV so the arrays are handled by exactly one code path.

Everything returns ``float32`` mono-ready arrays.  Nothing here guesses: if a file cannot
be decoded, :class:`~audio_preprocessing.exceptions.AudioRejected` is raised with a reason
code, and :func:`validate_file` reports the reason without raising so a batch upload can
list every bad file at once.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .config import audio_config, ffmpeg_binary, ffprobe_binary
from .config import quality_config as quality_settings
from .exceptions import (
    CORRUPT_HEADER,
    EMPTY_AUDIO,
    MISSING_FRAMES,
    NO_AUDIO_STREAM,
    NO_SUCH_FILE,
    SILENT,
    TOO_LARGE,
    TOO_LONG,
    TOO_SHORT,
    UNREADABLE_FORMAT,
    UNSUPPORTED_FORMAT,
    AudioRejected,
    AudioDecodeError,
)
from .transforms import amplitude_to_db

MAGIC_PREFIXES: tuple[tuple[bytes, str], ...] = (
    (b"RIFF", "wav"),
    (b"fLaC", "flac"),
    (b"OggS", "ogg"),
    (b"ID3", "mp3"),
    (b"\xff\xfb", "mp3"),
    (b"\xff\xf3", "mp3"),
    (b"\xff\xf2", "mp3"),
    (b"\xff\xe3", "mp3"),
)

EXTENSION_ALIASES = {
    "wave": "wav",
    "aiff": "aiff",
    "aif": "aiff",
    "m4a": "m4a",
    "mp4": "m4a",
    "mpeg": "mp3",
    "mpga": "mp3",
}

# Extensions we are willing to hand to a decoder.  This is *not* the configured supported
# list (that gate runs later, so an .aiff can be reported as "unsupported by configuration"
# rather than as damaged); it is the line between "an audio file we can attempt" and
# "not an audio file at all".  Without it a renamed text file becomes a decode failure and
# the user is told their file is corrupt when the real problem is that it is not audio.
DECODABLE_EXTENSIONS = frozenset({
    "wav", "wave", "mp3", "mpeg", "mpga", "flac", "ogg", "oga", "opus",
    "m4a", "mp4", "aac", "aiff", "aif", "wma", "amr", "au", "w64", "caf",
})

# Bytes we sniff to spot a damaged container before handing it to a decoder.
SNIFF_BYTES = 16


def _sniff_format(path: Path) -> str | None:
    """Guess a format from magic bytes.  ``m4a``/``mp4`` use a size-prefixed ``ftyp`` box."""
    try:
        with path.open("rb") as fh:
            head = fh.read(SNIFF_BYTES)
    except OSError:
        return None
    for magic, fmt in MAGIC_PREFIXES:
        if head.startswith(magic):
            return fmt
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        return "m4a" if brand in (b"M4A ", b"M4A\x20", b"mp42", b"isom", b"iso2", b"mp41", b"qt  ") else "mp4"
    return None


def detect_format(path: str | Path) -> str | None:
    """File format by content first, extension second.

    Content wins so that a ``.wav`` file which is really an MP3 is handled correctly
    instead of being reported as damaged.
    """
    p = Path(path)
    sniffed = _sniff_format(p)
    if sniffed is not None:
        return sniffed
    ext = p.suffix.lower().lstrip(".")
    return EXTENSION_ALIASES.get(ext, ext) or None


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash, used for FR lxxiii duplicate detection."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ffprobe_metadata(path: str | Path) -> dict[str, Any]:
    """Metadata for a file via ffprobe.  Returns ``{}`` when ffprobe is unavailable.

    Used for the things soundfile cannot tell us: the declared duration of a truncated
    file, the codec, and whether a container holds an audio stream at all.
    """
    binary = shutil.which(ffprobe_binary()) or ffprobe_binary()
    cmd = [
        binary, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        # ffprobe failed: that is a decode failure, not an absence of information.
        return {"_error": (proc.stderr or "ffprobe failed").strip()[:400]}
    import json

    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"_error": "ffprobe produced unparseable output"}


def _audio_streams(probe: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]


def _wave_metadata(path: Path) -> dict[str, Any]:
    """Bit depth and true frame count from the WAV header (FR x asks for bit depth)."""
    try:
        with wave.open(str(path), "rb") as wf:
            return {
                "sample_rate": wf.getframerate(),
                "channels": wf.getnchannels(),
                "n_frames": wf.getnframes(),
                "bit_depth": wf.getsampwidth() * 8,
                "duration_sec": wf.getnframes() / float(wf.getframerate() or 1),
                "compression": wf.getcomptype(),
            }
    except (wave.Error, OSError, EOFError):
        return {}


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------

def _decode_with_soundfile(path: Path, *, offset: float, duration: float | None) -> tuple[np.ndarray, int]:
    import soundfile as sf

    start = int(offset) if offset else 0
    frames = -1 if duration is None else max(1, int(round(duration * 16000)))
    data, sr = sf.read(
        str(path),
        start=start,
        frames=frames if offset else (-1 if duration is None else int(round(duration * 16000))),
        dtype="float32",
        always_2d=True,
    )
    return data, int(sr)


def _decode_with_ffmpeg(path: Path, *, offset: float, duration: float | None) -> tuple[np.ndarray, int]:
    """Decode anything ffmpeg can read into a temporary WAV, then load that."""
    import soundfile as sf

    binary = shutil.which(ffmpeg_binary()) or ffmpeg_binary()
    with tempfile.TemporaryDirectory(prefix="sonicsentinel_decode_") as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = [binary, "-v", "error", "-nostdin", "-y"]
        if offset:
            cmd += ["-ss", f"{offset:.6f}"]
        cmd += ["-i", str(path)]
        if duration is not None:
            cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-ac", "1", "-f", "wav", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        if proc.returncode != 0 or not out.exists():
            raise AudioDecodeError(
                (proc.stderr or "ffmpeg could not decode the file").strip()[:400]
            )
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return data, int(sr)


def load_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 array.

    Parameters
    ----------
    sample_rate:
        Target rate; ``None`` keeps the file's own rate (useful for metadata reporting).
    mono:
        Downmix multi-channel input by averaging channels (SRS Step 4).
    offset, duration:
        Decode only a slice -- this is how a live rolling buffer is read back.
    max_seconds:
        Refuse anything longer.  Applied as a hard read limit as well, so a 3-hour
        mislabelled upload cannot exhaust memory before the duration check runs.

    Returns
    -------
    (samples, sample_rate)
        ``samples`` is 2-D only when ``mono=False`` and the source is multi-channel.
        Otherwise it is a 1-D float32 array.

    Raises
    ------
    AudioRejected
        With a stable reason code, never a bare exception and never a silent empty array.
    """
    p = Path(path)
    if not p.exists():
        raise AudioRejected(NO_SUCH_FILE, f"no file at {p}")
    if not p.is_file():
        raise AudioRejected(NO_SUCH_FILE, f"{p} is not a regular file")
    if p.stat().st_size == 0:
        raise AudioRejected(EMPTY_AUDIO, "the file is zero bytes")

    read_duration = duration
    if max_seconds is not None:
        capped = max_seconds - max(0.0, offset)
        if capped <= 0:
            raise AudioRejected(TOO_SHORT, "requested slice starts beyond the analysis limit")
        read_duration = capped if read_duration is None else min(read_duration, capped)

    errors: list[str] = []
    data: np.ndarray | None = None
    sr: int | None = None

    try:
        data, sr = _decode_with_soundfile(p, offset=offset, duration=read_duration)
    except Exception as exc:  # soundfile raises several unrelated exception types
        errors.append(f"soundfile: {type(exc).__name__}: {exc}"[:300])

    if data is None or data.size == 0:
        try:
            data, sr = _decode_with_ffmpeg(p, offset=offset, duration=read_duration)
        except Exception as exc:
            errors.append(f"ffmpeg: {type(exc).__name__}: {exc}"[:300])
            detail = " | ".join(errors) if errors else "no decoder succeeded"
            raise AudioDecodeError(
                f"could not decode {p.name}: {detail}",
                metrics={"format": detect_format(p), "size_bytes": p.stat().st_size},
            ) from exc

    if data is None or data.size == 0:
        raise AudioRejected(EMPTY_AUDIO, f"{p.name} decoded to zero audio samples")
    if sr is None or sr <= 0:
        raise AudioRejected(UNREADABLE_FORMAT, f"{p.name} declares an invalid sample rate")

    y = np.asarray(data, dtype=np.float32)
    if mono:
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = np.ascontiguousarray(y.reshape(-1), dtype=np.float32)
        target = sample_rate
        if target is not None and int(target) != int(sr):
            from .transforms import resample

            y = resample(y, orig_sr=int(sr), target_sr=int(target))
            sr = int(target)
    else:
        if y.ndim == 1:
            y = y.reshape(-1, 1)

    finite = np.isfinite(y)
    if not finite.all():
        # A decoder producing NaN/Inf means damaged data.  Replace with zeros rather than
        # propagating NaN into every downstream feature, and record that we did.
        y = np.where(finite, y, np.float32(0.0))

    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def load_audio_bytes(
    data: bytes,
    *,
    filename: str = "upload",
    **kwargs: Any,
) -> tuple[np.ndarray, int]:
    """Load in-memory upload bytes.  The suffix is preserved so format detection works."""
    suffix = Path(filename).suffix or ".bin"
    with tempfile.TemporaryDirectory(prefix="sonicsentinel_upload_") as tmp:
        tmp_path = Path(tmp) / f"upload{suffix}"
        tmp_path.write_bytes(data)
        return load_audio(tmp_path, **kwargs)


# --------------------------------------------------------------------------------------
# Validation (SRS Step 3, FR viii)
# --------------------------------------------------------------------------------------

def validate_file(
    path: str | Path,
    *,
    cfg: dict[str, Any] | None = None,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a candidate upload.  Never raises; always explains itself.

    Returns a dict with:

    ``ok``        -- True only when the file is usable.
    ``reason``    -- a reason code from :mod:`audio_preprocessing.exceptions`, or ``None``.
    ``detail``    -- human-readable message.
    ``format``, ``duration_sec``, ``sample_rate``, ``channels``, ``bit_depth``,
    ``size_bytes``, ``codec``, ``n_frames`` -- the metadata FR x requires, where available.
    """
    settings = cfg or audio_config()
    supported = [str(s).lower() for s in settings.get("supported_formats", [])]
    max_mb = float(settings.get("max_upload_mb", 50))
    min_dur = float(settings.get("min_duration_sec", 0.5))
    max_dur = float(settings.get("max_duration_sec", 300.0))

    info: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "detail": "",
        "path": str(path),
        "format": None,
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "size_bytes": None,
        "codec": None,
        "n_frames": None,
    }

    p = Path(path)
    if not p.exists() or not p.is_file():
        info.update(reason=NO_SUCH_FILE, detail=f"no file at {p}")
        return info

    size = p.stat().st_size
    info["size_bytes"] = size
    info["format"] = detect_format(p)

    if size == 0:
        info.update(reason=EMPTY_AUDIO, detail="the file is zero bytes")
        return info
    if size > max_mb * 1024 * 1024:
        info.update(reason=TOO_LARGE, detail=f"{size / 1048576:.1f} MB exceeds the {max_mb:.0f} MB limit")
        return info

    # "Is this audio at all?" comes before "is the header readable?" -- magic bytes win over
    # the extension, so a WAV renamed to .txt is still validated as audio, while a text file
    # is reported as an unsupported format instead of as a damaged container.
    if _sniff_format(p) is None:
        fmt_ext = (info["format"] or "").lower()
        if fmt_ext not in DECODABLE_EXTENSIONS:
            info.update(
                reason=UNSUPPORTED_FORMAT,
                detail=(
                    f"'{fmt_ext or 'unknown'}' is not an audio format we can read; "
                    f"supported formats: {', '.join(supported) or 'none configured'}"
                ),
            )
            return info

    if info["format"] == "wav":
        wave_meta = _wave_metadata(p)
        if wave_meta:
            info["sample_rate"] = wave_meta.get("sample_rate")
            info["channels"] = wave_meta.get("channels")
            info["bit_depth"] = wave_meta.get("bit_depth")
            info["n_frames"] = wave_meta.get("n_frames")
            info["duration_sec"] = wave_meta.get("duration_sec")
            info["codec"] = "pcm"
        elif not probe:
            info.update(
                reason=CORRUPT_HEADER,
                detail="the file has a WAV header but the header could not be read",
            )
            return info

    if info["duration_sec"] is None or info["codec"] is None:
        probe_data = probe if probe is not None else ffprobe_metadata(p)
        streams = _audio_streams(probe_data)
        if probe_data.get("_error"):
            info.update(reason=CORRUPT_HEADER, detail=probe_data["_error"])
            return info
        if not streams:
            info.update(reason=NO_AUDIO_STREAM, detail="the file contains no audio track")
            return info
        stream = streams[0]
        info["codec"] = stream.get("codec_name")
        if stream.get("sample_rate"):
            info["sample_rate"] = int(stream["sample_rate"])
        if stream.get("channels"):
            info["channels"] = int(stream["channels"])
        if stream.get("duration"):
            try:
                info["duration_sec"] = float(stream["duration"])
            except (TypeError, ValueError):
                pass
        fmt = probe_data.get("format", {})
        if info["duration_sec"] is None and fmt.get("duration"):
            try:
                info["duration_sec"] = float(fmt["duration"])
            except (TypeError, ValueError):
                pass
        if info["n_frames"] is None and fmt.get("nb_streams") is None and stream.get("nb_frames"):
            try:
                info["n_frames"] = int(stream["nb_frames"])
            except (TypeError, ValueError):
                pass

    # Format gate: only after we know the file is otherwise real, so that a damaged file
    # is reported as damaged rather than as unsupported.
    fmt = (info["format"] or "").lower()
    if supported and fmt not in supported:
        info.update(
            reason=UNSUPPORTED_FORMAT,
            detail=f"format '{fmt}' is not in the supported list: {', '.join(supported)}",
        )
        return info

    # Duration gate.  Prefer the decoded truth: a truncated file's header lies.
    duration = info["duration_sec"]
    if duration is None:
        try:
            y, sr = load_audio(p, sample_rate=None, mono=True)
            duration = len(y) / float(sr or 1)
            info["duration_sec"] = duration
            info["sample_rate"] = info["sample_rate"] or sr
        except AudioRejected:
            info.update(reason=UNREADABLE_FORMAT, detail="the audio could not be decoded")
            return info

    if duration < min_dur:
        info.update(
            reason=TOO_SHORT,
            detail=f"{duration:.2f}s is shorter than the {min_dur:.2f}s minimum",
        )
        return info
    if duration > max_dur:
        info.update(
            reason=TOO_LONG,
            detail=f"{duration:.1f}s is longer than the {max_dur:.0f}s analysis limit",
        )
        return info

    # Integrity: does it actually decode, and is there a signal in it?
    try:
        y, _ = load_audio(p, sample_rate=None, mono=True)
    except AudioRejected as exc:
        info.update(reason=exc.reason, detail=exc.info.detail)
        return info

    if y.size == 0:
        info.update(reason=EMPTY_AUDIO, detail="decoded to zero samples")
        return info

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    if not np.isfinite(peak) or not np.isfinite(rms):
        info.update(reason=MISSING_FRAMES, detail="the decoded waveform contains non-finite values")
        return info

    info["peak"] = peak
    info["rms"] = rms

    # Presence of audio signal (FR viii, FR xii).  A silent recording is a valid container
    # holding no event, so it must be refused here and not discovered later as an
    # all-zeros prediction.
    silence_max = float(quality_settings().get("silence_rms_dbfs_max", -50.0))
    rms_db = amplitude_to_db(rms) if rms > 0 else -np.inf
    info["rms_dbfs"] = None if not np.isfinite(rms_db) else float(rms_db)
    if not np.isfinite(rms_db) or rms_db <= silence_max:
        info.update(
            reason=SILENT,
            detail=(
                f"the recording is silent: RMS "
                f"{'-inf' if not np.isfinite(rms_db) else f'{rms_db:.1f}'} dBFS is not above "
                f"the {silence_max:.0f} dBFS floor"
            ),
        )
        return info

    info["ok"] = True
    return info


def validate_samples(y: np.ndarray, sample_rate: int, *, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validation for live capture, where there is no file to inspect.

    Live windows are checked for presence of signal and for a sane duration, but not for
    format or container problems -- there is no container.
    """
    settings = cfg or audio_config()
    silence_max = float(quality_settings().get("silence_rms_dbfs_max", -50.0))
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    duration = arr.size / float(sample_rate or 1)
    floor = float(settings.get("min_duration_sec", 0.5))
    info: dict[str, Any] = {
        "ok": True,
        "reason": None,
        "detail": "",
        "format": "raw-live",
        "duration_sec": duration,
        "sample_rate": int(sample_rate),
        "channels": 1,
        "bit_depth": 32,
        "size_bytes": int(arr.size * 4),
        "codec": "pcm_f32le",
        "n_frames": int(arr.size),
    }
    if arr.size == 0:
        info.update(ok=False, reason=EMPTY_AUDIO, detail="the live window holds no samples")
        return info
    if duration < floor:
        info.update(
            ok=False,
            reason=TOO_SHORT,
            detail=f"live window of {duration:.2f}s is below the {floor:.2f}s minimum",
        )
        return info
    if not np.isfinite(arr).all():
        info.update(ok=False, reason=MISSING_FRAMES, detail="the live window contains non-finite samples")
        return info
    rms = float(np.sqrt(np.mean(np.square(arr, dtype=np.float64))))
    info["peak"] = float(np.max(np.abs(arr)))
    info["rms"] = rms
    # Presence of signal -- the docstring above promises this check, so it must exist.
    rms_db = amplitude_to_db(rms) if rms > 0 else -np.inf
    info["rms_dbfs"] = None if not np.isfinite(rms_db) else float(rms_db)
    if not np.isfinite(rms_db) or rms_db <= silence_max:
        info.update(
            ok=False,
            reason=SILENT,
            detail=f"the live window is silent (RMS {info['rms_dbfs']} dBFS)",
        )
    return info


def convert_format(
    src: str | Path,
    target_format: str,
    *,
    out_dir: str | Path | None = None,
    sample_rate: int | None = None,
    mono: bool = True,
    bit_depth: int = 16,
) -> Path:
    """Convert a file to another container/codec with ffmpeg (SRS Step 4 format conversion).

    Used by the dataset builder to normalise mixed-format source audio and by the tests to
    prove a round trip.  Raises :class:`AudioRejected` if ffmpeg cannot perform it.
    """
    src_path = Path(src)
    if not src_path.exists():
        raise AudioRejected(NO_SUCH_FILE, f"no file at {src_path}")
    fmt = str(target_format).lower().lstrip(".")
    # ``target_format`` may arrive as a filename (``Path("ok.flac")``) rather than a bare
    # extension; a stem must never be mistaken for the codec.
    if "." in Path(fmt).name:
        fmt = Path(fmt).suffix.lstrip(".")
    if fmt not in {"wav", "mp3", "flac", "ogg", "m4a", "aiff"}:
        raise AudioRejected(UNSUPPORTED_FORMAT, f"cannot convert to '{fmt}'")

    codecs = {
        "wav": ["-c:a", f"pcm_s{bit_depth}le"],
        "flac": ["-c:a", "flac"],
        "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "5"],
        "m4a": ["-c:a", "aac", "-b:a", "192k"],
        "aiff": ["-c:a", f"pcm_s{bit_depth}be"],
    }

    out_root = Path(out_dir) if out_dir is not None else src_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{src_path.stem}.{fmt}"
    if out_path.resolve() == src_path.resolve():
        out_path = out_root / f"{src_path.stem}_converted.{fmt}"

    binary = shutil.which(ffmpeg_binary()) or ffmpeg_binary()
    cmd = [binary, "-v", "error", "-nostdin", "-y", "-i", str(src_path)]
    if mono:
        cmd += ["-ac", "1"]
    if sample_rate is not None:
        cmd += ["-ar", str(int(sample_rate))]
    cmd += codecs[fmt] if fmt != "wav" else ["-c:a", f"pcm_s{bit_depth}le"]
    cmd += [str(out_path)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if proc.returncode != 0 or not out_path.exists():
        raise AudioRejected(
            UNREADABLE_FORMAT,
            f"ffmpeg could not convert to {fmt}: {(proc.stderr or '').strip()[:300]}",
        )
    return out_path
