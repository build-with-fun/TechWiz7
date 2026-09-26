"""One clip or live window in, one decision record out.

Uploads and live windows both go through AnalysisPipeline.analyse, so the two paths cannot
drift apart: preprocess and rate quality, score with the Python and Teachable Machine
models, compare, apply the class rule and repeated-detection confirmation, decide whether a
person must review it, then persist.

The TM model never sees the Python result. Both predictors receive the same
PreprocessedAudio and nothing else; the comparison is the first place their outputs meet.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Overridable by the app factory.
DEFAULT_PYTHON_MODEL_DIR = REPO_ROOT / "python_models" / "best"
DEFAULT_GTM_DIR = REPO_ROOT / "gtm_model"

#: Bump when the bit layout changes, so old stored fingerprints are not compared with new ones.
FINGERPRINT_VERSION = "perceptual-fp-1.0.0"

#: A frame counts as silence when it sits this far below the clip's own loudest frame.
SILENCE_DROP_DB = 25.0


class PipelineError(RuntimeError):
    """Anything that stops an analysis from producing a result."""


class ModelsUnavailable(PipelineError):
    """A model could not be loaded. Raised at start-up, never per request, naming the missing file.
    """


# Repeated-detection confirmation (FR xl, FR xlvi)

@dataclass
class Detection:
    """One window's opinion, as the confirmation counter sees it."""

    class_name: str
    at: float
    confidence: float
    agreed: bool
    quality: str
    source: str = "upload"


class RepeatTracker:
    """Counts consecutive qualifying detections of one class (FR xl, xlvi).

    A window that names a different class, or fails the agreement or quality gate, ends the
    streak. "Three in a row" is the rule; "three in the last eight seconds" could confirm a
    class from unrelated hits.
    """

    def __init__(self, thresholds: Mapping[str, Any], *, store: Any = None) -> None:
        cfg = dict(thresholds.get("repeat_detection", {}))
        self.needed = int(cfg.get("required_consecutive_detections", 3))
        self.window_seconds = float(cfg.get("window_seconds", 8.0))
        self.requires_agreement = bool(cfg.get("requires_model_agreement", True))
        self.min_quality = str(cfg.get("min_quality", "Acceptable"))
        self._store = store
        self._streaks: dict[str, list[Detection]] = {}
        self._confirmed: dict[str, float] = {}
        self._lock = threading.RLock()

    def _quality_ok(self, quality: str) -> bool:
        if self._store is None:
            return True
        try:
            return bool(self._store.quality_at_least(quality, self.min_quality))
        except Exception:  # unknown quality name: refuse to confirm rather than assume
            return False

    def qualifies(self, det: Detection) -> tuple[bool, str]:
        """Whether this window may contribute to a streak, and why not if it may not."""
        if self.requires_agreement and not det.agreed:
            return False, "the two models named different classes"
        if not self._quality_ok(det.quality):
            return False, f"audio quality '{det.quality}' is below {self.min_quality}"
        return True, ""

    def observe(
        self,
        *,
        class_name: str,
        agreed: bool,
        quality: str,
        confidence: float = 0.0,
        at: float | None = None,
        source: str = "upload",
        needed: int | None = None,
        min_quality: str | None = None,
        requires_agreement: bool | None = None,
    ) -> dict[str, Any]:
        """Record one window and return its confirmation state.

        ``confirmed`` stays set for the rest of the streak so the caller raises the alert once.
        """
        now = float(at if at is not None else time.time())
        det = Detection(class_name, now, float(confidence), bool(agreed), str(quality), source)
        required = max(1, int(needed if needed is not None else self.needed))
        agreement_required = (self.requires_agreement if requires_agreement is None
                              else bool(requires_agreement))
        quality_floor = min_quality or self.min_quality

        with self._lock:
            # Any streak for a different class is over: this is what "consecutive" means.
            for other in list(self._streaks):
                if other != class_name:
                    self._streaks.pop(other, None)
                    self._confirmed.pop(other, None)

            if agreement_required and not det.agreed:
                ok, why = False, "the two models named different classes"
            elif self._store is not None and not self._store.quality_at_least(det.quality, quality_floor):
                ok, why = False, f"audio quality '{det.quality}' is below {quality_floor}"
            else:
                ok, why = True, ""
            if not ok:
                # A disqualifying window does not merely fail to add -- it breaks the streak.
                self._streaks.pop(class_name, None)
                self._confirmed.pop(class_name, None)
                return {
                    "consecutive": 0,
                    "needed": required,
                    "confirmed": False,
                    "newly_confirmed": False,
                    "window_seconds": self.window_seconds,
                    "qualifies": False,
                    "note": f"Not counted towards confirmation: {why}.",
                }

            streak = self._streaks.setdefault(class_name, [])
            if streak and (now - streak[-1].at) > self.window_seconds:
                streak.clear()  # the gap was too long to call these detections consecutive
                self._confirmed.pop(class_name, None)
            streak.append(det)

            consecutive = len(streak)
            already = self._confirmed.get(class_name) is not None
            confirmed = already or consecutive >= required
            newly_confirmed = confirmed and not already
            if confirmed and not already:
                self._confirmed[class_name] = now

            if already:
                note = (
                    f"{class_name} was already confirmed after {required} consecutive "
                    "detections; the alert for this streak is not raised twice."
                )
            elif confirmed:
                note = (
                    f"Confirmed after {consecutive} consecutive detections within "
                    f"{self.window_seconds:.0f} s (required {required})."
                )
                streak.clear()  # start counting for a fresh alert after this one
            else:
                note = (
                    f"{consecutive} of {required} consecutive detections; "
                    f"{required - consecutive} more needed within "
                    f"{self.window_seconds:.0f} s."
                )

            return {
                "consecutive": consecutive,
                "needed": required,
                "confirmed": bool(confirmed),
                "newly_confirmed": bool(newly_confirmed),
                "window_seconds": self.window_seconds,
                "qualifies": True,
                "note": note,
            }

    def reset(self) -> None:
        with self._lock:
            self._streaks.clear()
            self._confirmed.clear()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "needed": self.needed,
                "window_seconds": self.window_seconds,
                "requires_model_agreement": self.requires_agreement,
                "min_quality": self.min_quality,
                "streaks": {k: len(v) for k, v in self._streaks.items()},
                "confirmed": sorted(self._confirmed),
            }


# Perceptual fingerprint (pipeline step 4, FR lxxiv)

def audio_fingerprint(samples: Any, sample_rate: int, *, bands: int = 16,
                      frame_ms: float = 64.0, max_frames: int = 32) -> str | None:
    """Coarse perceptual fingerprint used to shortlist near-duplicates (FR lxxiv).

    Band energies on a fixed grid, each compared with its band's median and reduced to one
    bit, so a quieter copy or a re-encode keeps most bits. On real recordings it is too coarse
    to decide alone (short impulses collide), so confirm_near_duplicate re-checks the shortlist
    with spectral_match. Returns hex, or None when the clip is too short.
    """
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        return None

    # A coarse magnitude spectrogram. Small n_fft keeps this cheap enough to sit inside a
    # 3 s live-window budget; the fingerprint compares shapes, not detail.
    n_fft = 1024
    hop = max(1, int(sample_rate * frame_ms / 1000.0))
    if y.size < n_fft * 2:
        return None
    window = np.hanning(n_fft)
    frames = []
    for start in range(0, y.size - n_fft + 1, hop):
        frame = y[start:start + n_fft] * window
        frames.append(np.abs(np.fft.rfft(frame)) ** 2)
    if len(frames) < 2:
        return None
    power = np.asarray(frames)

    # Collapse the frequency axis into mel-spaced bands: energy concentrated where a human
    # hears it, rather than where FFT bins happen to fall.
    edges = np.unique(np.floor(
        n_fft / 2 * (np.linspace(0, 1, bands + 1) ** 2)  # ~mel spacing, cheaply
    ).astype(int))
    banded_db = np.zeros((power.shape[0], max(len(edges) - 1, 1)))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        banded_db[:, i] = power[:, lo:hi].mean(axis=1)
    # dB, so "25 dB below the loudest frame" means the same thing for every clip.
    banded_db = 10.0 * np.log10(banded_db + 1e-12)

    # Fingerprint the *sound*, not the silence around it: trim head and tail frames more than
    # ``SILENCE_DROP_DB`` below this clip's loudest frame. Without this, the same recording
    # with a second of padding -- which any recorder, and any re-share, may add -- shifts the
    # whole time axis and reads as a different sound.
    frame_level = banded_db.max(axis=1)
    loud = frame_level >= (frame_level.max() - SILENCE_DROP_DB)
    active = np.flatnonzero(loud)
    if active.size >= 2:
        banded_db = banded_db[active[0]:active[-1] + 1]

    # Resample the time axis to a fixed length so a 2.9 s and a 3.1 s recording of the same
    # event still line up.
    if banded_db.shape[0] != max_frames:
        idx = np.linspace(0, banded_db.shape[0] - 1, max_frames)
        resampled = np.empty((max_frames, banded_db.shape[1]))
        for col in range(banded_db.shape[1]):
            resampled[:, col] = np.interp(idx, np.arange(banded_db.shape[0]), banded_db[:, col])
        banded_db = resampled
    banded = banded_db

    # Compare each cell with its band's median: robust to level changes and spectral tilt.
    # The band set must depend on the parameters only. An earlier version dropped
    # flat bands per clip, so two copies of one sound got different layouts and
    # scored near chance.
    median = np.median(banded, axis=0, keepdims=True)
    bits_array = (banded > median).astype(np.uint8).reshape(-1)
    if bits_array.size % 4:
        bits_array = np.concatenate([bits_array, np.zeros(4 - bits_array.size % 4, np.uint8)])
    packed = bits_array.reshape(-1, 4) @ np.array([8, 4, 2, 1], dtype=np.uint8)
    return np.asarray(packed, dtype=np.uint8).tobytes().hex()


def fingerprint_similarity(a: str | None, b: str | None) -> float | None:
    """Fraction of fingerprint bits two clips share, or ``None`` if either is missing."""
    if not a or not b or len(a) != len(b):
        return None
    try:
        left = np.frombuffer(bytes.fromhex(a), dtype=np.uint8)
        right = np.frombuffer(bytes.fromhex(b), dtype=np.uint8)
    except ValueError:
        return None
    bits_left = np.unpackbits(left)
    bits_right = np.unpackbits(right)
    if bits_left.size == 0:
        return None
    return float((bits_left == bits_right).mean())


def _source_sha256(path: str | Path | None) -> str | None:
    """sha256 of the clip's bytes on disk -- the exact-duplicate key in pipeline step 3."""
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


#: Fallback when config/thresholds.json has no duplicate_detection block.
DEFAULT_NEAR_DUPLICATE_SIMILARITY = 0.92


def duplicate_config(thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The duplicate-detection settings, with documented defaults if config is silent."""
    block = ((thresholds or {}).get("duplicate_detection") or {})
    return {
        "enabled": bool(block.get("enabled", True)),
        "near_duplicate_similarity": float(
            block.get("near_duplicate_similarity", DEFAULT_NEAR_DUPLICATE_SIMILARITY)
        ),
        "fingerprint_version": str(block.get("fingerprint_version", FINGERPRINT_VERSION)),
        # Two-stage check (26 Sep): the fingerprint shortlists, spectral_match decides.
        # Measured in reports/near_duplicates.json: stage-1-only flagged 19% of hard
        # negatives; shortlist 5 + spectral match >= 0.9 flagged 0.6%.
        "shortlist": int(block.get("shortlist", 5)),
        "spectral_match_min": float(block.get("spectral_match_min", 0.9)),
        "source": "config/thresholds.json" if block else "built-in default (config has no duplicate_detection block)",
    }


def find_near_duplicate(
    fingerprint: str | None,
    candidates: Any,
    *,
    threshold: float = DEFAULT_NEAR_DUPLICATE_SIMILARITY,
    exclude: Any = (),
) -> dict[str, Any] | None:
    """Closest candidate at or above ``threshold`` by fingerprint alone.

    Used when the caller has no stored audio to confirm against.
    """
    if not fingerprint:
        return None
    skip = set(exclude)
    best: dict[str, Any] | None = None
    for audio_id, other in candidates:
        if audio_id in skip or not other:
            continue
        similarity = fingerprint_similarity(fingerprint, other)
        if similarity is None or similarity < threshold:
            continue
        if best is None or similarity > best["similarity"]:
            best = {"audio_id": audio_id, "similarity": round(similarity, 6), "threshold": threshold}
    return best


def _log_mel_frames(samples: Any, sample_rate: int) -> np.ndarray:
    import librosa

    y = np.asarray(samples, dtype=np.float32).ravel()
    if sample_rate != 16000:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=320, n_mels=40)
    logmel = np.log(mel + 1e-6).T                       # (frames, bands), 20 ms per frame
    # Subtracting the clip's own mean makes a volume-adjusted copy identical in this
    # representation: a gain is a constant added to every log-mel cell.
    return logmel - logmel.mean()


def spectral_match(a: Any, sr_a: int, b: Any, sr_b: int) -> float:
    """Best-aligned Pearson correlation of two log-mel images (1.0 means the same sound).

    The shorter clip slides along the longer one, so a trimmed copy lines up with its source.
    """
    left, right = _log_mel_frames(a, sr_a), _log_mel_frames(b, sr_b)
    short, long_ = (left, right) if len(left) <= len(right) else (right, left)
    n = len(short)
    if n < 10:
        return 0.0
    flat = (short - short.mean()).ravel()
    flat_norm = float(np.linalg.norm(flat)) or 1.0

    def corr(offset: int) -> float:
        window = long_[offset:offset + n]
        w = (window - window.mean()).ravel()
        return float(flat @ w) / (flat_norm * (float(np.linalg.norm(w)) or 1.0))

    offsets = range(0, len(long_) - n + 1)
    coarse = max(offsets[::5], key=corr)                 # 100 ms steps, then refine
    return max(corr(o) for o in range(max(0, coarse - 5), min(len(long_) - n, coarse + 5) + 1))


def confirm_near_duplicate(
    samples: Any,
    sample_rate: int,
    fingerprint: str | None,
    candidates: Any,
    *,
    shortlist: int = 5,
    min_match: float = 0.9,
    exclude: Any = (),
    load: Callable[[Any], tuple[Any, int]] | None = None,
) -> dict[str, Any] | None:
    """Shortlist candidates by fingerprint, then confirm the shortlist with spectral_match.

    ``load`` must condition the stored file the way ``samples`` was conditioned: comparing a
    preprocessed upload with a raw stored file scored an identical sound 0.86 instead of 1.0.
    Candidates whose file has gone (retention) are skipped.
    """
    if not fingerprint:
        return None
    if load is None:
        import librosa

        def load(path):
            return librosa.load(str(path), sr=16000, mono=True, duration=60.0)

    skip = set(exclude)
    ranked = sorted(
        ((fingerprint_similarity(fingerprint, fp) or 0.0, audio_id, path)
         for audio_id, fp, path in candidates if audio_id not in skip and fp and path),
        reverse=True,
    )[:shortlist]
    best: dict[str, Any] | None = None
    for similarity, audio_id, path in ranked:
        try:
            other, rate = load(path)
        except Exception:  # noqa: BLE001 - an unreadable stored file just isn't compared
            continue
        match = spectral_match(samples, sample_rate, other, rate)
        if match >= min_match and (best is None or match > best["spectral_match"]):
            best = {"audio_id": audio_id, "similarity": round(similarity, 6),
                    "spectral_match": round(match, 4), "threshold": min_match}
    return best


# Severity + alert eligibility (Step 16, FR lii-lv)

def severity_block(
    class_name: str,
    *,
    store: Any,
    confidence: float,
    top_two_margin: float,
    quality: str,
    classes_agree: bool,
    critical: bool | None = None,
    consecutive: int = 0,
    noise_level_dbfs: float | None = None,
) -> dict[str, Any]:
    """Severity for the event record, and whether it may raise an alert.

    Severity is recorded for every detection. Alert eligibility is a separate set of gates
    (confidence, margin, agreement, quality, minimum alert severity): a Gunshot at 0.61 where
    the models disagree is stored as a Gunshot event but pages nobody.
    """
    rule = store.rule_for_class(class_name)
    recorded = str(rule.get("severity", "Informational"))
    shown = store.display_severity(recorded)
    is_critical = store.is_critical(class_name) if critical is None else bool(critical)

    gates: list[dict[str, Any]] = []

    def gate(name: str, required: Any, observed: Any, ok: bool) -> None:
        gates.append({"gate": name, "required": required, "observed": observed, "passed": bool(ok)})

    if rule.get("min_confidence") is not None:
        req = float(rule["min_confidence"])
        gate("min_confidence", req, round(float(confidence), 6), float(confidence) >= req)
    if rule.get("min_top_two_margin") is not None:
        req = float(rule["min_top_two_margin"])
        gate("min_top_two_margin", req, round(float(top_two_margin), 6),
             float(top_two_margin) >= req)
    if rule.get("requires_model_agreement") is not None:
        gate("requires_model_agreement", bool(rule["requires_model_agreement"]),
             bool(classes_agree), (not bool(rule["requires_model_agreement"])) or bool(classes_agree))
    if rule.get("min_audio_quality") is not None:
        minimum = str(rule["min_audio_quality"])
        try:
            ok = bool(store.quality_at_least(str(quality), minimum))
        except Exception:
            ok = False
        gate("min_audio_quality", minimum, str(quality), ok)

    gate("enabled", True, bool(rule.get("enabled", True)), bool(rule.get("enabled", True)))
    if class_name == "Background Noise":
        limit = float((rule.get("thresholds") or {}).get("noise_level_dbfs_limit", -35.0))
        gate("noise_level_dbfs", limit, noise_level_dbfs,
             noise_level_dbfs is not None and noise_level_dbfs >= limit)

    eligible = all(g["passed"] for g in gates) if gates else True

    matched_escalation = None
    for escalation in rule.get("escalation", []):
        conditions = escalation.get("when") or {}
        observed = {
            "confidence": confidence,
            "consecutive": consecutive,
            "top_two_margin": top_two_margin,
            "noise_level_dbfs": noise_level_dbfs,
            "quality": quality,
        }
        if _escalation_matches(conditions, observed):
            matched_escalation = escalation.get("id")
            recorded = escalation.get("to_severity", recorded)
            shown = store.display_severity(recorded)
            rule = {**rule, "recommended_action": escalation.get("to_action", rule.get("recommended_action"))}
            break

    # FR xliv/xlv: a vehicle horn or an animal is stored as an event, not raised as an
    # alert. Checked after escalation, so loud background noise escalated to Medium
    # (FR l) still alerts. The floor comes from the rules file ("min_alert_severity").
    floor = rule.get("min_alert_severity")
    if floor:
        ok = store.severity_rank(recorded) >= store.severity_rank(str(floor))
        gate("min_alert_severity", str(floor), recorded, ok)
        eligible = eligible and ok

    return {
        "recorded": recorded,
        "severity": recorded,
        "severity_display": shown,
        "severity_rank": store.severity_rank(recorded),
        "critical_class": is_critical,
        "recommended_action": str(rule.get("recommended_action", "")),
        "alert_eligible": eligible,
        "gates": gates,
        "srs_ref": rule.get("srs_ref"),
        "rule_class": rule.get("class", class_name),
        "matched_escalation": matched_escalation,
    }


def _escalation_matches(conditions: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    """Evaluate only conditions backed by measurements in this decision record."""
    if "all" in conditions:
        return all(_escalation_matches(item, observed) for item in conditions["all"])
    fields = {
        "confidence_gte": "confidence",
        "consecutive_gte": "consecutive",
        "top_two_margin_gte": "top_two_margin",
        "noise_level_dbfs_gte": "noise_level_dbfs",
    }
    for key, required in conditions.items():
        if key == "quality_in":
            if observed.get("quality") not in required:
                return False
        elif key in fields:
            value = observed.get(fields[key])
            if value is None or float(value) < float(required):
                return False
        else:
            return False
    return bool(conditions)


# Manual review (Step 17, FR lvii, li, xxxviii, xxxix)

class _SafeDict(dict):
    """format_map helper: an unknown template key renders as {key} instead of crashing a live
    window.
    """

    def __missing__(self, key: str) -> str:  # pragma: no cover - exercised via tests
        return "{" + key + "}"


def _template_context(ctx: Mapping[str, Any]) -> _SafeDict:
    safe = _SafeDict()
    for key, value in ctx.items():
        if isinstance(value, float):
            safe[key] = round(value, 6)
        elif isinstance(value, (int, str, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _eval_predicate(when: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    """Evaluate one ``when`` clause from manual_review_conditions.json.

    The vocabulary is closed on purpose: an administrator-editable condition language that
    could run code would be remote code execution. Unknown keys fail closed.
    """
    for key, expected in when.items():
        if key == "any":
            if not any(_eval_predicate(sub, ctx) for sub in expected):
                return False
        elif key == "all":
            if not all(_eval_predicate(sub, ctx) for sub in expected):
                return False
        elif key == "models_agree":
            if bool(ctx.get("models_agree")) != bool(expected):
                return False
        elif key == "overlapping":
            if bool(ctx.get("overlapping")) != bool(expected):
                return False
        elif key == "unknown_class":
            if bool(ctx.get("unknown_class")) != bool(expected):
                return False
        elif key == "below_min_confidence_on_both":
            if bool(ctx.get("below_min_confidence_on_both")) != bool(expected):
                return False
        elif key == "class_is_critical":
            if bool(ctx.get("class_is_critical")) != bool(expected):
                return False
        elif key == "any_model_confidence_lt":
            if not (float(ctx.get("python_confidence", 1.0)) < float(expected)
                    or float(ctx.get("gtm_confidence", 1.0)) < float(expected)):
                return False
        elif key == "top_two_margin_lt":
            if not float(ctx.get("top_two_margin", 1.0)) < float(expected):
                return False
        elif key == "confidence_difference_gt":
            if not float(ctx.get("confidence_difference", 0.0)) > float(expected):
                return False
        elif key == "quality_in":
            if str(ctx.get("quality")) not in [str(q) for q in expected]:
                return False
        elif key == "consistency_status_in":
            if str(ctx.get("consistency_status")) not in [str(s) for s in expected]:
                return False
        else:  # unknown key: a condition we cannot evaluate must not silently pass
            raise PipelineError(
                f"manual_review_conditions.json uses an unknown condition '{key}'. "
                "Refusing to run: a review gate that is not evaluated is worse than none."
            )
    return True


def evaluate_review(
    *,
    store: Any,
    python_class: str,
    python_confidence: float,
    gtm_class: str,
    gtm_confidence: float,
    classes_agree: bool,
    confidence_difference: float,
    top_two_margin: float,
    consistency_status: str,
    quality: str,
    overlapping: bool,
    secondary_detection: str | None = None,
    predicted_class: str | None = None,
    reviewed_class: str | None = None,
    near_duplicate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Which review conditions fired, each with its own reason and recommended action."""
    thresholds = store.thresholds()
    primary = predicted_class or python_class
    known = set(store.class_names())
    ctx: dict[str, Any] = {
        "python_class": python_class,
        "python_confidence": float(python_confidence),
        "gtm_class": gtm_class,
        "gtm_confidence": float(gtm_confidence),
        "models_agree": bool(classes_agree),
        "confidence_difference": float(confidence_difference),
        "top_two_margin": float(top_two_margin),
        "margin": float(top_two_margin),
        "consistency_status": str(consistency_status),
        "quality": str(quality),
        "quality_detail": str(quality),
        "overlapping": bool(overlapping),
        "secondary_detection": secondary_detection or "",
        "predicted_class": primary,
        "class_is_critical": bool(store.is_critical(primary)),
        "unknown_class": primary not in known and bool(primary),
        "below_min_confidence_on_both": (
            float(python_confidence) < float(thresholds["confidence"]["min_confidence"])
            and float(gtm_confidence) < float(thresholds["confidence"]["min_confidence"])
        ),
        "min_confidence": float(thresholds["confidence"]["min_confidence"]),
        "low_confidence_band": float(
            thresholds["confidence"].get(
                "low_confidence_band", thresholds["confidence"]["min_confidence"]
            )
        ),
        "margin_min": float(thresholds["confidence"]["top_two_margin_min"]),
        "reviewed_class": reviewed_class or primary,
        "near_duplicate": bool(near_duplicate),
    }

    matched: list[dict[str, Any]] = []
    for condition in store.review_conditions():
        when = condition.get("when") or {}
        if _eval_predicate(when, ctx):
            reason = str(condition.get("reason_template", "")).format_map(_template_context(ctx))
            matched.append(
                {
                    "id": condition.get("id"),
                    "priority": condition.get("priority", "normal"),
                    "srs_phrase": condition.get("srs_phrase"),
                    "reason": reason,
                    "recommended_action": condition.get("recommended_action", ""),
                }
            )

    # FR lxxiv: a likely copy is surfaced for review, never merged or dropped. A condition
    # with id near_duplicate in manual_review_conditions.json overrides this fallback.
    if near_duplicate and "near_duplicate" not in {str(m["id"]) for m in matched}:
        configured = next(
            (c for c in store.review_conditions() if str(c.get("id")) == "near_duplicate"), None
        )
        audio_id = near_duplicate.get("audio_id")
        similarity = float(near_duplicate.get("similarity", 0.0))
        if configured:
            fallback_ctx = dict(ctx)
            fallback_ctx.update({"near_duplicate_of": audio_id, "similarity": similarity})
            matched.append(
                {
                    "id": "near_duplicate",
                    "priority": configured.get("priority", "normal"),
                    "srs_phrase": configured.get("srs_phrase"),
                    "reason": str(configured.get("reason_template", "")).format_map(
                        _template_context(fallback_ctx)
                    ),
                    "recommended_action": configured.get("recommended_action", ""),
                }
            )
        else:
            matched.append(
                {
                    "id": "near_duplicate",
                    "priority": "normal",
                    "srs_phrase": "Near-Duplicate Detection (FR lxxiv)",
                    "reason": (
                        f"This recording is {similarity:.0%} similar to an existing analysis "
                        f"({audio_id}) -- it looks like a re-encoded, trimmed or "
                        "volume-adjusted copy rather than a new event."
                    ),
                    "recommended_action": (
                        "Confirm whether this is the same event as the earlier recording "
                        "before counting it as a new detection."
                    ),
                    "fallback": True,
                }
            )

    queue = store.manual_review().get("queue", {})
    priority_rank = {"critical": 3, "high": 2, "normal": 1, "low": 0}
    top_priority = max(
        (str(m["priority"]) for m in matched),
        key=lambda p: priority_rank.get(p, 0),
        default=str(queue.get("default_priority", "normal")),
    )

    clean = (
        bool(classes_agree)
        and consistency_status == "Strong Match"
        and quality == "Good"
        and float(python_confidence) >= ctx["low_confidence_band"]
        and float(gtm_confidence) >= ctx["low_confidence_band"]
        and float(top_two_margin) >= ctx["margin_min"]
        # A clip flagged as a probable re-share is not a "clean result" even if the models
        # are certain: the question on the table is provenance, not accuracy.
        and not near_duplicate
    )

    return {
        "required": bool(matched),
        "matched": [m["id"] for m in matched],
        "findings": matched,
        "priority": top_priority if matched else None,
        "conditions_evaluated": len(store.review_conditions()),
        "clean_result": clean,
        # A clean result with no firing condition is the only shape that stays out of the queue.
        "never_auto_review_ok": (not clean) or not matched,
    }


# The pipeline

def _extractor_for_bundle(model_dir: Path) -> Any:
    """The feature extractor the saved bundle was trained with, from its feature_config.json."""
    import json as _json

    from feature_extraction.features import FeatureExtractor

    config_path = model_dir / "feature_config.json"
    version = ""
    if config_path.exists():
        version = str(_json.loads(config_path.read_text(encoding="utf-8")).get("feature_version", ""))
    if version.startswith("panns"):
        from feature_extraction import embeddings as backbone

        extractor_cls = backbone.EmbeddingFeatureExtractor
    elif version.startswith("ast"):
        from feature_extraction import ast_embeddings as backbone

        extractor_cls = backbone.AstFeatureExtractor
    else:
        return FeatureExtractor()
    if version != backbone.EMBEDDING_VERSION:
        raise ModelsUnavailable(
            f"{model_dir} was trained on embeddings {version!r} but this checkout "
            f"produces {backbone.EMBEDDING_VERSION!r}; retrain or check out the matching code"
        )
    return extractor_cls()


@dataclass
class PipelineModels:
    """The two independent models, held together only so they are loaded once."""

    python: Any  # src.inference.predictor.PythonModelPredictor
    gtm: Any  # src.inference.gtm_predictor.GtmModelPredictor
    preprocessor: Any  # audio_preprocessing.pipeline.AudioPipeline
    feature_extractor: Any  # feature_extraction.features.FeatureExtractor

    def describe(self) -> dict[str, Any]:
        describe = lambda obj: obj.describe() if hasattr(obj, "describe") else {}
        return {
            "python": describe(self.python),
            "gtm": describe(self.gtm),
            "preprocessing": describe(self.preprocessor),
            "features": describe(self.feature_extractor),
        }


class AnalysisPipeline:
    """Runs one analysis and returns the decision record.

    ``persist`` is optional so the pipeline can be tested and timed without a database.
    """

    def __init__(
        self,
        models: PipelineModels,
        *,
        store: Any = None,
        tracker: RepeatTracker | None = None,
        persist: Callable[[dict[str, Any]], Any] | None = None,
        model_dir: str | Path | None = None,
        gtm_dir: str | Path | None = None,
    ) -> None:
        from src.services.config import get_store

        self.models = models
        self.store = store if store is not None else get_store()
        self.tracker = tracker if tracker is not None else RepeatTracker(
            self.store.thresholds(), store=self.store
        )
        self.persist = persist
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_PYTHON_MODEL_DIR
        self.gtm_dir = Path(gtm_dir) if gtm_dir else DEFAULT_GTM_DIR
        self._warmed: dict[str, float] | None = None

    # construction
    @classmethod
    def load(
        cls,
        *,
        model_dir: str | Path | None = None,
        gtm_dir: str | Path | None = None,
        store: Any = None,
        persist: Callable[[dict[str, Any]], Any] | None = None,
    ) -> "AnalysisPipeline":
        """Load both models, the preprocessor and the extractor, or raise ModelsUnavailable.

        There is deliberately no single-model fallback.
        """
        from audio_preprocessing.pipeline import AudioPipeline
        from src.inference.gtm_predictor import GtmModelPredictor
        from src.inference.predictor import PythonModelPredictor
        from src.services.config import ConfigError, get_store

        if store is None:
            try:
                store = get_store()
                problems = store.validate()
                if problems:
                    raise PipelineError(
                        "configuration is invalid:\n  - " + "\n  - ".join(problems)
                    )
            except ConfigError as exc:  # pragma: no cover - misconfigured checkout
                raise ModelsUnavailable(f"configuration could not be loaded: {exc}") from exc

        thresholds = store.thresholds()
        model_dir = Path(model_dir) if model_dir else DEFAULT_PYTHON_MODEL_DIR
        gtm_dir = Path(gtm_dir) if gtm_dir else DEFAULT_GTM_DIR

        preprocessor = AudioPipeline()

        if not (Path(model_dir) / "model.joblib").exists():
            raise ModelsUnavailable(
                f"no saved Python model at {model_dir}/model.joblib. "
                "Train and export the Python classifier before starting analysis."
            )
        try:
            extractor = _extractor_for_bundle(Path(model_dir))
            python_predictor = PythonModelPredictor.load(model_dir, extractor.extract)
        except Exception as exc:
            raise ModelsUnavailable(
                f"the saved Python model at {model_dir} could not be loaded: {exc}"
            ) from exc

        if not Path(gtm_dir).exists():
            raise ModelsUnavailable(
                f"no Google Teachable Machine export at {gtm_dir}. "
                "Install the separate Teachable Machine audio export and frontend "
                "configuration before starting analysis."
            )
        try:
            gtm_predictor = GtmModelPredictor.load(gtm_dir)
        except Exception as exc:
            raise ModelsUnavailable(
                f"the Teachable Machine model at {gtm_dir} could not be loaded: {exc}"
            ) from exc

        # The two models must name the same ten classes in the same terms, or the
        # comparison report would be comparing different questions.
        py_classes = list(python_predictor.class_names)
        gtm_classes = list(gtm_predictor.class_names)
        wanted = store.class_names()
        drift = {
            "python": sorted(set(py_classes) ^ set(wanted)),
            "gtm": sorted(set(gtm_classes) ^ set(wanted)),
        }
        if any(drift.values()):
            raise ModelsUnavailable(
                f"class vocabulary drift against config/classes.json: {drift}. "
                f"config/classes.json defines {wanted}."
            )

        return cls(
            PipelineModels(
                python=python_predictor,
                gtm=gtm_predictor,
                preprocessor=preprocessor,
                feature_extractor=extractor,
            ),
            store=store,
            tracker=RepeatTracker(thresholds, store=store),
            persist=persist,
            model_dir=model_dir,
            gtm_dir=gtm_dir,
        )

    # start-up
    def warm(self) -> dict[str, Any]:
        """Pay librosa/numba and first-model-call costs at start-up rather than in the first
        request.
        """
        from audio_preprocessing.pipeline import warm_up

        started = time.perf_counter()
        timings = warm_up(int(self.store.thresholds()["audio"]["target_sample_rate"]))

        model_warm_ms: dict[str, float] = {}
        try:
            import numpy as np

            seconds = float(self.store.thresholds()["audio"].get("segment_duration_sec", 3.0))
            sr = int(self.store.thresholds()["audio"]["target_sample_rate"])
            probe = np.zeros(int(sr * seconds), dtype=np.float32)
            t0 = time.perf_counter()
            pre = self.models.preprocessor.preprocess_samples(probe, sr)
            model_warm_ms["preprocess"] = round((time.perf_counter() - t0) * 1000.0, 3)
            if not pre.rejected:
                t0 = time.perf_counter()
                self.models.feature_extractor.extract(pre)
                model_warm_ms["features"] = round((time.perf_counter() - t0) * 1000.0, 3)
                t0 = time.perf_counter()
                self.models.python.predict_from_preprocessed(pre)
                model_warm_ms["python_predict"] = round((time.perf_counter() - t0) * 1000.0, 3)
                t0 = time.perf_counter()
                self.models.gtm.predict_from_preprocessed(pre)
                model_warm_ms["gtm_predict"] = round((time.perf_counter() - t0) * 1000.0, 3)
        except Exception as exc:  # a warm-up failure must not stop the app from starting
            model_warm_ms["error"] = f"{type(exc).__name__}: {exc}"

        self._warmed = {**timings, **model_warm_ms}
        self._warmed["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return dict(self._warmed)

    @property
    def warmed(self) -> dict[str, float] | None:
        return dict(self._warmed) if self._warmed else None

    # the analysis
    def analyse(
        self,
        source: Any,
        *,
        origin: str = "upload",
        meta: Mapping[str, Any] | None = None,
        at: float | None = None,
        persist: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Classify one AudioSource and return the full decision record.

        Bad audio never raises: a rejected clip comes back with ``ok=False``, the reason and
        whatever quality measurements exist, so the UI can show it. ``meta`` is stored but never
        affects a prediction.
        """
        started = time.perf_counter()
        meta = dict(meta or {})
        thresholds = self.store.thresholds()
        budgets = thresholds.get("performance", {})
        if origin == "live":
            budget_sec = float(budgets.get("max_live_window_budget", 3.0))
        else:
            budget_sec = float(budgets.get("max_upload_seconds_budget", 8.0))

        timings: dict[str, float] = {}

        def mark(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000.0, 3)
            return now

        record: dict[str, Any] = {
            "ok": False,
            "status": "rejected",
            "origin": origin,
            "meta": meta,
            "created_at": _iso(at if at is not None else time.time()),
            "budget_sec": budget_sec,
        }

        # Preprocessing also produces the validation result and the quality verdict.
        t0 = time.perf_counter()
        try:
            preprocessed = self.models.preprocessor(source)
        except Exception as exc:
            from audio_preprocessing.exceptions import AudioRejected, message_for

            mark("preprocess", t0)
            code = getattr(exc, "reason", None) or type(exc).__name__
            detail = ""
            metrics: dict[str, Any] = {}
            if isinstance(exc, AudioRejected):
                info = getattr(exc, "info", None)
                detail = getattr(info, "detail", "") or ""
                metrics = dict(getattr(info, "metrics", None) or {})
            record["rejection"] = {
                "code": code,
                "reason": code,
                "detail": detail,
                "message": message_for(code) if isinstance(exc, AudioRejected) else str(exc),
                "stage": "decode",
                "metrics": metrics,
            }
            if isinstance(exc, AudioRejected):
                record["quality"] = {
                    "verdict": "Unusable",
                    "problems": [code],
                    "urgent": [code],
                    "summary": message_for(code),
                    "notes": [],
                    "measurements": metrics,
                    "thresholds": {},
                }
            record["timings_ms"] = timings
            return self._finish(record, started)
        mark("preprocess", t0)

        record["preprocessing"] = {
            "version": (preprocessed.preprocessing or {}).get("version"),
            "steps": (preprocessed.preprocessing or {}).get("steps", []),
            "source_origin": (preprocessed.preprocessing or {}).get("source_origin", origin),
            "config_source": (preprocessed.preprocessing or {}).get("config_source"),
        }
        record["audio"] = {
            "duration_sec": round(float(preprocessed.duration_sec), 4),
            "sample_rate": int(preprocessed.sample_rate),
            # Source rate/channels: the report shows what was received as well as what was analysed.
            "source_sample_rate": (preprocessed.preprocessing or {}).get("source_sample_rate"),
            "source_channels": (preprocessed.preprocessing or {}).get("source_channels"),
            "source_bit_depth": (preprocessed.preprocessing or {}).get("source_bit_depth"),
            "n_segments": len(preprocessed.segments or []),
            "source_path": str(preprocessed.source_path) if preprocessed.source_path else None,
            # sha256 identifies the bytes, the fingerprint the sound. The exact-duplicate 409 needs
            # the database, so the caller does it.
            "sha256": meta.get("sha256") or _source_sha256(preprocessed.source_path),
            "fingerprint": audio_fingerprint(
                preprocessed.samples,
                int(preprocessed.sample_rate or (self.store.thresholds().get("audio", {})
                                                 .get("target_sample_rate", 16000))),
            ),
            "fingerprint_version": FINGERPRINT_VERSION,
        }
        quality = dict(preprocessed.quality or {})
        record["quality"] = {
            "verdict": quality.get("verdict"),
            "problems": list(quality.get("problems", []) or []),
            "urgent": list(quality.get("urgent", []) or []),
            "summary": quality.get("summary"),
            "notes": list(quality.get("notes", []) or []),
            "measurements": dict(quality.get("measurements", {}) or {}),
            "thresholds": dict(quality.get("thresholds", {}) or {}),
        }

        if preprocessed.rejected:
            rejection = dict((preprocessed.preprocessing or {}).get("rejection", {}) or {})
            from audio_preprocessing.exceptions import message_for

            reason = rejection.get("reason") or "unusable_audio"
            record["rejection"] = {
                "code": reason,
                "reason": reason,
                "detail": rejection.get("detail", ""),
                "message": preprocessed.rejection_reason or message_for(reason),
                "stage": "preprocess",
                "quality_verdict": quality.get("verdict"),
                "quality_problems": list(quality.get("problems", []) or []),
            }
            record["status"] = "rejected"
            record["timings_ms"] = timings
            return self._finish(record, started)

        # Both models get `preprocessed` and nothing else.
        t0 = time.perf_counter()
        t_py, t_gtm = t0, t0
        try:
            py_result = self.models.python.predict_from_preprocessed(preprocessed, origin=origin)
            t_py = mark("python_model", t0)
        except Exception as exc:
            record["error"] = {
                "stage": "python_model",
                "message": f"{type(exc).__name__}: {exc}",
            }
            record["status"] = "error"
            record["timings_ms"] = timings
            return self._finish(record, started)

        t0 = t_py
        try:
            gtm_result = self.models.gtm.predict_from_preprocessed(preprocessed, origin=origin)
            t_gtm = mark("gtm_model", t0)
        except Exception as exc:
            record["error"] = {
                "stage": "gtm_model",
                "message": f"{type(exc).__name__}: {exc}",
            }
            record["status"] = "error"
            record["predictions"] = {"python": _prediction(py_result)}
            record["timings_ms"] = timings
            return self._finish(record, started)

        record["predictions"] = {
            "python": _prediction(py_result),
            "gtm": _prediction(gtm_result),
        }

        t0 = t_gtm
        from src.inference.consistency import classify_consistency, requires_manual_review

        comparison = classify_consistency(py_result, gtm_result, thresholds)
        comparison_dict = comparison.to_dict()
        mark("compare", t0)
        record["comparison"] = comparison_dict
        # The older one-line verdict is kept next to the condition-based one; they answer
        # slightly different questions.
        legacy_required, legacy_reason = requires_manual_review(comparison, thresholds)
        record["comparison"]["requires_review_quick_check"] = bool(legacy_required)
        record["comparison"]["quick_check_reason"] = legacy_reason

        # Severity, alert eligibility and repeated-detection confirmation.
        t0 = time.perf_counter()
        primary = py_result.predicted_class  # the Python model drives the decision; the
        #                                       GTM result grades it (SRS Step 11)
        rule = self.store.rule_for_class(primary)
        confirmation = self.tracker.observe(
            class_name=primary,
            agreed=bool(comparison.classes_agree),
            quality=str(record["quality"].get("verdict")),
            confidence=float(py_result.confidence),
            at=at,
            source=origin,
            # A live stream can wait for the next window; an uploaded recording is finished,
            # so it is confirmed by the other SRS gates (agreement and quality here,
            # confidence and margin in severity_block) with required_for_uploads windows.
            needed=(int((thresholds.get("repeat_detection") or {}).get("required_for_uploads", 1))
                    if origin == "upload" else rule.get("required_consecutive_detections")),
            min_quality=rule.get("min_audio_quality"),
            requires_agreement=rule.get("requires_model_agreement"),
        )
        noise_level = (record["quality"].get("measurements") or {}).get("rms_dbfs")
        severity = severity_block(
            primary,
            store=self.store,
            confidence=float(py_result.confidence),
            top_two_margin=float(comparison.top_two_margin_python),
            quality=str(record["quality"].get("verdict")),
            classes_agree=bool(comparison.classes_agree),
            consecutive=confirmation["consecutive"],
            noise_level_dbfs=noise_level,
        )
        alert = {
            "raised": bool(severity["alert_eligible"] and confirmation["newly_confirmed"]),
            "eligible": bool(severity["alert_eligible"]),
            "confirmed": bool(confirmation["confirmed"]),
            "consecutive": confirmation["consecutive"],
            "needed": confirmation["needed"],
            "window_seconds": confirmation["window_seconds"],
            "note": confirmation["note"],
            "requires_acknowledgement": bool(
                severity["critical_class"] and severity["alert_eligible"]
            ),
            "acknowledged": False,
            "recommended_action": severity["recommended_action"],
        }
        record["severity"] = severity
        record["alert"] = alert
        mark("rules", t0)

        t0 = time.perf_counter()
        # Candidates come from the caller (the database); without them nothing is claimed.
        duplicate_settings = duplicate_config(thresholds)
        near_duplicate = None
        candidates = list(meta.get("near_duplicate_candidates") or ())
        exclude = {meta.get("audio_id")} if meta.get("audio_id") else ()
        if duplicate_settings["enabled"] and record.get("audio", {}).get("fingerprint"):
            if candidates and len(candidates[0]) == 3:
                # The upload route passes stored file paths, so the stronger two-stage
                # check can run; callers with fingerprints only get stage 1.
                near_duplicate = confirm_near_duplicate(
                    preprocessed.samples, int(preprocessed.sample_rate),
                    record["audio"]["fingerprint"], candidates,
                    shortlist=duplicate_settings["shortlist"],
                    min_match=duplicate_settings["spectral_match_min"], exclude=exclude,
                    load=self._preprocessed_file,
                )
            else:
                near_duplicate = find_near_duplicate(
                    record["audio"]["fingerprint"], candidates,
                    threshold=duplicate_settings["near_duplicate_similarity"], exclude=exclude,
                )
        record["duplicate"] = {
            "sha256": record.get("audio", {}).get("sha256"),
            "exact_duplicate_of": meta.get("exact_duplicate_of"),
            # None means "we did not find a match", which is a different statement from
            # "we did not look" -- `checked` says which.
            "near_duplicate_of": (near_duplicate or {}).get("audio_id"),
            "near_duplicate_similarity": (near_duplicate or {}).get("similarity"),
            "near_duplicate_threshold": duplicate_settings["near_duplicate_similarity"],
            "fingerprint_version": record.get("audio", {}).get("fingerprint_version"),
            "candidates_compared": len(candidates),
            "near_duplicate_spectral_match": (near_duplicate or {}).get("spectral_match"),
            "checked": bool(duplicate_settings["enabled"] and record.get("audio", {}).get("fingerprint")),
            "settings_source": duplicate_settings["source"],
        }
        review = evaluate_review(
            store=self.store,
            python_class=py_result.predicted_class,
            python_confidence=float(py_result.confidence),
            gtm_class=gtm_result.predicted_class,
            gtm_confidence=float(gtm_result.confidence),
            classes_agree=bool(comparison.classes_agree),
            confidence_difference=float(comparison.confidence_difference),
            top_two_margin=float(comparison.top_two_margin_python),
            consistency_status=str(comparison.consistency_status),
            quality=str(record["quality"].get("verdict")),
            overlapping=bool(comparison.overlapping),
            secondary_detection=comparison.secondary_detection,
            predicted_class=primary,
            near_duplicate=near_duplicate,
        )
        # A critical class detected without agreement must be reviewed even if no
        # condition happened to name it -- the queue is the safety net for exactly that.
        if severity["critical_class"] and not comparison.classes_agree and not review["required"]:
            review["required"] = True
            review["matched"] = list(review["matched"]) + ["critical_manual_review"]
            review["findings"] = list(review["findings"]) + [
                {
                    "id": "critical_manual_review",
                    "priority": "critical",
                    "srs_phrase": "Critical class detected without model agreement",
                    "reason": (
                        f"{primary} is a critical class and the two models did not agree "
                        f"({comparison.consistency_status})."
                    ),
                    "recommended_action": "Review immediately before acting on the alert.",
                }
            ]
            review["priority"] = "critical"
        record["review"] = review
        mark("review", t0)

        record["ok"] = True
        record["status"] = "analysed"
        record["model_versions"] = {
            "python": {
                "name": py_result.model_name,
                "version": py_result.model_version,
                "feature_version": py_result.feature_version,
            },
            "gtm": {
                "name": gtm_result.model_name,
                "version": gtm_result.model_version,
                "feature_version": gtm_result.feature_version,
            },
        }
        record["config_snapshot"] = self.store.snapshot().to_dict()
        record["decision"] = {
            "final_class": primary,
            "final_decision": _final_decision(primary, review),
            "severity_display": severity["severity_display"],
            "alert_raised": alert["raised"],
            "review_required": review["required"],
            "review_priority": review.get("priority"),
        }

        callback = persist if persist is not None else self.persist
        if callback is not None:
            t0 = time.perf_counter()
            try:
                stored = callback(record)
                if isinstance(stored, Mapping):
                    record["stored"] = dict(stored)
                    if stored.get("event_ids"):
                        record["event_id"] = stored["event_ids"][0]
                    record["audio_id"] = stored.get("audio_id")
                    record["alert_id"] = stored.get("alert_id")
                    record["review_id"] = stored.get("review_id")
                elif stored is not None:
                    record["stored"] = {"event_id": stored}
                    record["event_id"] = stored
            except Exception as exc:
                # Losing the write must not lose the analysis the user is waiting for; it is
                # recorded so the failure is visible instead of silently dropped.
                record["stored"] = {"error": f"{type(exc).__name__}: {exc}"}
            mark("persist", t0)

        record["timings_ms"] = timings
        return self._finish(record, started)

    def _preprocessed_file(self, path: Any) -> tuple[Any, int]:
        pre = self.models.preprocessor.preprocess_file(path)
        if getattr(pre, "rejected", False):
            raise ValueError(pre.rejection_reason)
        return pre.samples, int(pre.sample_rate)

    def analyse_file(
        self, path: str | Path, *, origin: str = "upload",
        persist: Callable[[dict[str, Any]], Any] | None = None, **meta: Any,
    ) -> dict[str, Any]:
        from src.inference.contract import AudioSource

        path = Path(path)
        payload = dict(meta)
        payload.setdefault("filename", path.name)
        return self.analyse(AudioSource.from_path(path, origin=origin), origin=origin,
                            meta=payload, persist=persist)

    def analyse_bytes(
        self,
        data: bytes,
        *,
        filename: str = "upload.wav",
        origin: str = "upload",
        meta: Mapping[str, Any] | None = None,
        persist: Callable[[dict[str, Any]], Any] | None = None,
        **extra_meta: Any,
    ) -> dict[str, Any]:
        """Analyse uploaded bytes (written to a content-addressed temp file so FFmpeg can read it).
        """
        from src.inference.contract import AudioSource

        tmp_dir = REPO_ROOT / "data" / "tmp" / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Content-addressed name: two uploads of the same bytes cannot collide, and purge
        # by prefix stays simple.
        digest = hashlib.sha256(data).hexdigest()[:16]
        suffix = Path(filename).suffix or ".wav"
        tmp_path = tmp_dir / f"{digest}{suffix}"
        if not tmp_path.exists():
            tmp_path.write_bytes(data)
        payload = dict(meta or {})
        payload.update(extra_meta)
        payload.setdefault("filename", filename)
        payload["sha256"] = hashlib.sha256(data).hexdigest()
        return self.analyse(AudioSource.from_path(tmp_path, origin=origin), origin=origin,
                            meta=payload, persist=persist)

    def analyse_samples(
        self,
        samples: Any,
        sample_rate: int,
        *,
        origin: str = "live",
        persist: Callable[[dict[str, Any]], Any] | None = None,
        **meta: Any,
    ) -> dict[str, Any]:
        """Analyse one already-captured window (the live microphone path)."""
        from src.inference.contract import AudioSource

        return self.analyse(
            AudioSource.from_samples(samples, sample_rate, origin=origin),
            origin=origin,
            meta=dict(meta),
            persist=persist,
        )

    def _finish(self, record: dict[str, Any], started: float) -> dict[str, Any]:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        record["elapsed_ms"] = elapsed_ms
        record["within_budget"] = elapsed_ms <= float(record.get("budget_sec", 0.0)) * 1000.0
        if not record["within_budget"]:
            record["budget_overrun_ms"] = round(
                elapsed_ms - float(record["budget_sec"]) * 1000.0, 3
            )
        record.setdefault("timings_ms", {})
        record["timings_ms"]["total"] = elapsed_ms
        return record


def _prediction(result: Any) -> dict[str, Any]:
    """One model's output, with every class's confidence -- the UI shows all ten."""
    return {
        "model_name": result.model_name,
        "model_version": result.model_version,
        "predicted_class": result.predicted_class,
        "confidence": round(float(result.confidence), 6),
        "confidences": {k: round(float(v), 6) for k, v in dict(result.confidences).items()},
        "top3": [{"class": c, "confidence": round(float(v), 6)} for c, v in result.top_k(3)],
        "latency_sec": round(float(result.latency_sec), 5),
        "feature_version": result.feature_version,
        "source_origin": result.source_origin,
    }


def _final_decision(primary: str, review: Mapping[str, Any]) -> str:
    """The wording the UI shows: Likely Valid / Likely Invalid / Manual Review Required."""
    if review.get("required"):
        return "Manual Review Required"
    return "Likely Valid"


def _iso(epoch: float) -> str:
    import datetime as _dt

    return (
        _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# Process-wide pipeline (the app factory builds it once; endpoints ask for it)

_PIPELINE: AnalysisPipeline | None = None
_PIPELINE_LOCK = threading.RLock()


def get_pipeline() -> AnalysisPipeline:
    if _PIPELINE is None:
        raise PipelineError(
            "the analysis pipeline has not been initialised. The app factory must call "
            "set_pipeline(AnalysisPipeline.load(...)) at start-up."
        )
    return _PIPELINE


def set_pipeline(pipeline: AnalysisPipeline | None) -> None:
    global _PIPELINE
    with _PIPELINE_LOCK:
        _PIPELINE = pipeline


def pipeline_status() -> dict[str, Any]:
    """What ``/api/health`` and the admin dashboard report about the live pipeline."""
    if _PIPELINE is None:
        return {"ready": False, "reason": "pipeline not initialised"}
    return {
        "ready": True,
        "models": _PIPELINE.models.describe(),
        "warmed": _PIPELINE.warmed,
        "tracker": _PIPELINE.tracker.state(),
        "config": _PIPELINE.store.snapshot().to_dict(),
    }
