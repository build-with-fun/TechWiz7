"""Audio quality verdict (SRS Step 13, FR xii-xiv): Good, Acceptable, Poor or Unusable.

Runs the seven SRS checks (silence, clipping, noise, low signal, duration, encoding
problems, missing frames). A Poor clip can't confirm an alert on its own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import quality_config
from .transforms import peak_dbfs, rms_dbfs, to_mono

GOOD = "Good"
ACCEPTABLE = "Acceptable"
POOR = "Poor"
UNUSABLE = "Unusable"

QUALITY_LEVELS = (UNUSABLE, POOR, ACCEPTABLE, GOOD)
QUALITY_ORDER = {name: i for i, name in enumerate(QUALITY_LEVELS)}

EPS = 1e-12
FRAME_LENGTH = 2048
HOP_LENGTH = 512


def meets_min_quality(verdict: str, minimum: str) -> bool:
    """True if ``verdict`` is at least ``minimum``."""
    return QUALITY_ORDER.get(verdict, -1) >= QUALITY_ORDER.get(minimum, 99)


def estimate_snr_db(
    y: np.ndarray,
    sample_rate: int | None = None,
    *,
    n_fft: int = 1024,
    hop_length: int = 512,
    signal_fraction: float = 0.10,
    max_db: float = 120.0,
) -> float:
    """SNR estimate in dB: strongest spectral bins against the median of the rest.

    We don't use the quietest frames as the noise estimate because a steady siren or hum has
    no quiet frames (a clean tone scored about -23 dB that way).
    """
    arr = to_mono(y)
    if arr.size == 0:
        return 0.0

    import librosa

    if arr.size < n_fft:
        arr = np.pad(arr, (0, n_fft - arr.size))
    stft = librosa.stft(arr, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    bin_power = np.mean(np.abs(stft) ** 2, axis=1)
    n_bins = bin_power.size
    if n_bins < 4 or float(np.max(bin_power)) <= EPS:
        return 0.0

    k = max(1, int(round(float(signal_fraction) * n_bins)))
    k = min(k, n_bins - 1)
    order = np.argsort(bin_power)
    floor_bins = order[: n_bins - k]
    peak_bins = order[n_bins - k:]
    floor_power = float(np.median(bin_power[floor_bins]))
    peak_power = float(np.mean(bin_power[peak_bins]))

    if floor_power <= EPS:
        # Clean synthetic signal: cap instead of returning inf.
        return float(max_db) if peak_power > EPS else 0.0
    if peak_power <= floor_power:
        return 0.0
    return float(min(max_db, 10.0 * np.log10(peak_power / floor_power)))


def frame_dynamic_range_db(
    y: np.ndarray,
    *,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
    low: float = 10.0,
    high: float = 90.0,
) -> float:
    """Range between quiet and loud frames in dB. Reported only, not used in the verdict."""
    frames = _frame_powers(to_mono(y), frame_length=frame_length, hop_length=hop_length)
    if frames.size == 0 or float(np.max(frames)) <= EPS:
        return 0.0
    lo = float(np.percentile(frames, low))
    hi = float(np.percentile(frames, high))
    if lo <= EPS:
        return float(min(120.0, 10.0 * np.log10(hi / EPS))) if hi > EPS else 0.0
    if hi <= lo:
        return 0.0
    return float(min(120.0, 10.0 * np.log10(hi / lo)))


def _frame_powers(y: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    """Mean power per frame, using a strided view."""
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size < frame_length:
        return np.array([float(np.mean(np.square(arr)))]) if arr.size else np.array([])
    n_frames = 1 + (arr.size - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return np.mean(np.square(arr[idx]), axis=1)


def clipping_stats(y: np.ndarray, threshold: float = 0.99) -> dict[str, float]:
    """Clipping ratio and the longest run at full scale (real clipping gives long runs)."""
    arr = to_mono(y)
    if arr.size == 0:
        return {"clipping_ratio": 0.0, "clipped_samples": 0, "longest_run": 0, "peak_dbfs": float("-inf")}
    flag = np.abs(arr) >= threshold
    clipped = int(np.count_nonzero(flag))
    longest = 0
    if clipped:
        padded = np.concatenate(([False], flag, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        longest = int(np.max(ends - starts)) if starts.size else 0
    return {
        "clipping_ratio": float(clipped / arr.size),
        "clipped_samples": clipped,
        "longest_run": longest,
        "peak_dbfs": peak_dbfs(arr),
    }


def analyze_quality(
    y: np.ndarray,
    sample_rate: int,
    *,
    cfg: dict[str, Any] | None = None,
    decoded_with_error: bool = False,
    non_finite_count: int | None = None,
) -> dict[str, Any]:
    """Run the Step 13 checks and return the verdict, problems, measurements and notes."""
    settings = cfg or quality_config()
    arr = to_mono(y)
    n = int(arr.size)

    # NaN/Inf samples are decoder gaps: zero them and count them.
    detected_missing = int(np.count_nonzero(~np.isfinite(arr))) if n else 0
    if detected_missing:
        arr = np.where(np.isfinite(arr), arr, np.float32(0.0)).astype(np.float32)
    duration = n / float(sample_rate) if sample_rate else 0.0

    checks: dict[str, bool] = {}
    notes: list[str] = []
    measurements: dict[str, Any] = {
        "duration_sec": round(duration, 6),
        "samples": n,
        "sample_rate": int(sample_rate),
    }

    # 1. signal presence / silence
    rms_db = rms_dbfs(arr)
    peak_db = peak_dbfs(arr)
    measurements["rms_dbfs"] = _clean(rms_db)
    measurements["peak_dbfs"] = _clean(peak_db)
    silence_max = float(settings.get("silence_rms_dbfs_max", -50.0))
    # Silence is judged on the loudest 50 ms frame, not the whole-clip RMS, which made a
    # short gunshot followed by quiet look silent.
    frame = max(1, int(0.05 * sample_rate)) if sample_rate else max(1, n)
    usable = (n // frame) * frame
    if usable:
        frames = arr[:usable].reshape(-1, frame).astype(np.float64)
        loudest = float(np.sqrt(np.max(np.mean(frames ** 2, axis=1))))
        loudest_db = 20.0 * np.log10(loudest) if loudest > 0 else float("-inf")
    else:
        loudest_db = rms_db
    measurements["loudest_frame_rms_dbfs"] = _clean(loudest_db)
    checks["silence"] = bool(n == 0 or loudest_db <= silence_max)
    if checks["silence"]:
        notes.append(f"loudest 50 ms frame {_fmt_db(loudest_db)} is at or below the silence "
                     f"floor {silence_max:.1f} dBFS")

    # 2. clipping
    clip = clipping_stats(arr)
    measurements.update(
        clipping_ratio=round(clip["clipping_ratio"], 6),
        clipped_samples=clip["clipped_samples"],
        longest_clip_run=clip["longest_run"],
    )
    clip_max = float(settings.get("clipping_ratio_max", 0.01))
    checks["clipping"] = bool(clip["clipping_ratio"] > clip_max)
    if checks["clipping"]:
        notes.append(
            f"{clip['clipping_ratio'] * 100:.2f}% of samples are clipped "
            f"(limit {clip_max * 100:.2f}%), longest full-scale run {clip['longest_run']} samples"
        )

    # 3. excessive noise / SNR
    snr_db = estimate_snr_db(arr)
    measurements["snr_db"] = _clean(snr_db)
    good_snr = float(settings.get("good_snr_db", 20.0))
    acceptable_snr = float(settings.get("acceptable_snr_db", 12.0))
    poor_snr = float(settings.get("poor_snr_db", 6.0))
    min_snr = float(settings.get("min_snr_db", 5.0))
    checks["excessive_noise"] = bool(snr_db < poor_snr)
    checks["noise_below_acceptable"] = bool(snr_db < acceptable_snr)
    if checks["excessive_noise"]:
        notes.append(f"SNR {_fmt_db(snr_db)} is below the poor-noise floor {poor_snr:.1f} dB")
    elif checks["noise_below_acceptable"]:
        notes.append(f"SNR {_fmt_db(snr_db)} is below the acceptable floor {acceptable_snr:.1f} dB")

    # 4. low signal strength (close to the quantisation floor)
    low_peak = float(settings.get("low_signal_peak_dbfs", -40.0))
    checks["low_signal"] = bool(not checks["silence"] and peak_db < low_peak)
    measurements["low_signal_peak_dbfs"] = low_peak
    if checks["low_signal"]:
        notes.append(f"peak {_fmt_db(peak_db)} is below the low-signal floor {low_peak:.1f} dBFS")

    # 5. duration
    min_dur = float(settings.get("min_duration_sec", 0.5))
    max_dur = float(settings.get("max_duration_sec", 300.0))
    checks["duration_short"] = bool(duration < min_dur)
    checks["duration_long"] = bool(duration > max_dur)
    if checks["duration_short"]:
        notes.append(f"duration {duration:.2f}s is below the {min_dur:.2f}s minimum")
    if checks["duration_long"]:
        notes.append(f"duration {duration:.1f}s exceeds the {max_dur:.0f}s limit")

    # 6. encoding problems
    checks["encoding"] = bool(decoded_with_error)
    if checks["encoding"]:
        notes.append("the decoder reported an encoding problem while reading this file")

    # 7. missing frames
    missing = max(int(non_finite_count or 0), detected_missing)
    measurements["non_finite_samples"] = missing
    checks["missing_frames"] = bool(missing > 0)
    if checks["missing_frames"]:
        notes.append(f"{missing} sample(s) could not be decoded (missing frames)")

    # verdict
    verdict, urgent, problems = _decide(
        checks, snr_db=snr_db, peak_db=peak_db,
        min_snr=min_snr, poor_snr=poor_snr, acceptable_snr=acceptable_snr, good_snr=good_snr,
        duration=duration, min_dur=min_dur,
        clip=clip, clip_max=clip_max,
    )

    return {
        "verdict": verdict,
        "problems": problems,
        "urgent": urgent,
        "checks": checks,
        "measurements": measurements,
        "thresholds": {
            "silence_rms_dbfs_max": silence_max,
            "clipping_ratio_max": clip_max,
            "min_snr_db": min_snr,
            "poor_snr_db": poor_snr,
            "acceptable_snr_db": acceptable_snr,
            "good_snr_db": good_snr,
            "min_duration_sec": min_dur,
            "max_duration_sec": max_dur,
        },
        "notes": notes,
        "config_source": settings.get("_source"),
    }


def _decide(
    checks: dict[str, bool],
    *,
    snr_db: float,
    peak_db: float,
    min_snr: float,
    poor_snr: float,
    acceptable_snr: float,
    good_snr: float,
    duration: float,
    min_dur: float,
    clip: dict[str, float],
    clip_max: float,
) -> tuple[str, list[str], list[str]]:
    """Measurements -> verdict. Only ``urgent`` problems make a clip Unusable."""
    problems: list[str] = []
    urgent: list[str] = []

    if checks.get("silence"):
        problems.append("silence")
        urgent.append("silence")
    if checks.get("missing_frames"):
        problems.append("missing_frames")
    if checks.get("encoding"):
        problems.append("encoding")
    if checks.get("duration_short"):
        problems.append("unsuitable_duration")
        urgent.append("unsuitable_duration")
    if checks.get("duration_long"):
        problems.append("unsuitable_duration")
    if checks.get("clipping"):
        problems.append("clipping")
    if checks.get("excessive_noise"):
        problems.append("excessive_noise")
    if checks.get("low_signal"):
        problems.append("low_signal")

    if urgent:
        return UNUSABLE, urgent, problems

    # More than 5x the clipping tolerance.
    severe_clip = checks.get("clipping") and clip["clipping_ratio"] > 5.0 * clip_max
    very_short = duration < max(min_dur, 0.6 * min_dur + 0.2) and duration < min_dur * 1.5
    # A faint event is still analysed, rated Poor, and goes to a reviewer.
    if severe_clip or snr_db < poor_snr or checks.get("duration_long") or very_short or checks.get("low_signal"):
        return POOR, urgent, problems
    # Decoder gaps are never Acceptable.
    if checks.get("missing_frames"):
        return POOR, urgent, problems

    if checks.get("clipping") or checks.get("noise_below_acceptable") or checks.get("encoding"):
        return ACCEPTABLE, urgent, problems

    healthy = (
        snr_db >= good_snr
        and peak_db > -12.0
        and duration >= max(min_dur, 1.0)
        and not checks.get("clipping")
        and not checks.get("missing_frames")
    )
    return (GOOD if healthy else ACCEPTABLE), urgent, problems


def _clean(value: float) -> float | str:
    """Make a dB value JSON-safe (no inf or nan)."""
    if value is None:
        return "n/a"
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(float(value), 4)


def _fmt_db(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if value == float("inf"):
        return "no measurable noise floor"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.1f} dB"


def quality_summary(quality: dict[str, Any]) -> str:
    """One-line summary for the UI and report."""
    verdict = quality.get("verdict", "Unknown")
    problems = quality.get("problems") or []
    if not problems:
        return f"Audio quality {verdict}: no problems detected."
    return f"Audio quality {verdict}: {', '.join(p.replace('_', ' ') for p in problems)}."
