"""Analysis pipeline: one clip or live window in, one decision record out.

Uploads and live windows share AnalysisPipeline.analyse: preprocess, rate quality, run
both models, compare them, apply the alert and review rules, then save. The two models
get the same preprocessed audio and never see each other's output.
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

# The app factory can override these.
DEFAULT_PYTHON_MODEL_DIR = REPO_ROOT / "python_models" / "best"
DEFAULT_GTM_DIR = REPO_ROOT / "gtm_model"

# Bump when the bit layout changes so old fingerprints are not compared with new ones.
FINGERPRINT_VERSION = "perceptual-fp-1.0.0"

# Frames this far below the clip's loudest frame count as silence.
SILENCE_DROP_DB = 25.0


class PipelineError(RuntimeError):
    """Anything that stops an analysis from producing a result."""


class ModelsUnavailable(PipelineError):
    """A model could not be loaded. Raised at start-up, not per request."""


# Repeated-detection confirmation (FR xl, FR xlvi)

@dataclass
class Detection:
    """One window's result, as seen by RepeatTracker."""

    class_name: str
    at: float
    confidence: float
    agreed: bool
    quality: str
    source: str = "upload"


class RepeatTracker:
    """Counts consecutive qualifying detections of one class (FR xl, xlvi).

    A different class, or a window that fails the agreement or quality check, ends the streak.
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
        except Exception:  # unknown quality name: don't confirm
            return False

    def qualifies(self, det: Detection) -> tuple[bool, str]:
        """Return (ok, reason) for whether this window can count toward a streak."""
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

        ``confirmed`` stays set for the rest of the streak, so the alert is raised once.
        """
        now = float(at if at is not None else time.time())
        det = Detection(class_name, now, float(confidence), bool(agreed), str(quality), source)
        required = max(1, int(needed if needed is not None else self.needed))
        agreement_required = (self.requires_agreement if requires_agreement is None
                              else bool(requires_agreement))
        quality_floor = min_quality or self.min_quality

        with self._lock:
            # A new class ends every other streak.
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
                # A window that doesn't qualify breaks the streak.
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
                streak.clear()  # gap too long to count as consecutive
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
                streak.clear()  # start over for the next alert
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
    """Coarse perceptual fingerprint for shortlisting near-duplicates (FR lxxiv).

    Each cell of a band-energy grid becomes one bit (above or below its band's median), so a
    quieter copy or a re-encode keeps most bits. Too coarse to decide on its own, so
    confirm_near_duplicate checks the shortlist with spectral_match. Returns hex, or None
    when the clip is too short.
    """
    y = np.asarray(samples, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        return None

    # Small n_fft keeps this cheap enough for the 3 s live-window budget.
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

    # Group the FFT bins into roughly mel-spaced bands.
    edges = np.unique(np.floor(
        n_fft / 2 * (np.linspace(0, 1, bands + 1) ** 2)  # ~mel spacing, cheaply
    ).astype(int))
    banded_db = np.zeros((power.shape[0], max(len(edges) - 1, 1)))
    for i in range(len(edges) - 1):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        banded_db[:, i] = power[:, lo:hi].mean(axis=1)
    # Work in dB so the silence threshold means the same for every clip.
    banded_db = 10.0 * np.log10(banded_db + 1e-12)

    # Trim leading and trailing silence, otherwise the same recording with a second of
    # padding shifts the time axis and looks like a different sound.
    frame_level = banded_db.max(axis=1)
    loud = frame_level >= (frame_level.max() - SILENCE_DROP_DB)
    active = np.flatnonzero(loud)
    if active.size >= 2:
        banded_db = banded_db[active[0]:active[-1] + 1]

    # Fixed number of time steps so 2.9 s and 3.1 s copies still line up.
    if banded_db.shape[0] != max_frames:
        idx = np.linspace(0, banded_db.shape[0] - 1, max_frames)
        resampled = np.empty((max_frames, banded_db.shape[1]))
        for col in range(banded_db.shape[1]):
            resampled[:, col] = np.interp(idx, np.arange(banded_db.shape[0]), banded_db[:, col])
        banded_db = resampled
    banded = banded_db

    # Comparing with the band median ignores level changes and spectral tilt. The band
    # layout must not depend on the clip, or two copies end up with different layouts.
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
    """sha256 of the file, used for exact-duplicate detection."""
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


# Used when config/thresholds.json has no duplicate_detection block.
DEFAULT_NEAR_DUPLICATE_SIMILARITY = 0.92


def duplicate_config(thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Duplicate-detection settings, falling back to the defaults above."""
    block = ((thresholds or {}).get("duplicate_detection") or {})
    return {
        "enabled": bool(block.get("enabled", True)),
        "near_duplicate_similarity": float(
            block.get("near_duplicate_similarity", DEFAULT_NEAR_DUPLICATE_SIMILARITY)
        ),
        "fingerprint_version": str(block.get("fingerprint_version", FINGERPRINT_VERSION)),
        # Fingerprint shortlists, spectral_match decides. On hard negatives this cut false
        # flags from 19% to 0.6% (reports/near_duplicates.json).
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
    # In log-mel a gain change is a constant offset, so removing the mean ignores volume.
    return logmel - logmel.mean()


def spectral_match(a: Any, sr_a: int, b: Any, sr_b: int) -> float:
    """Best-aligned correlation of two log-mel images (1.0 means the same sound).

    The shorter clip slides along the longer one so a trimmed copy still matches.
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
    """Shortlist candidates by fingerprint, then confirm them with spectral_match.

    ``load`` must preprocess the stored file the same way as ``samples``, otherwise an
    identical sound scores about 0.86. Candidates whose file was purged are skipped.
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
        except Exception:  # noqa: BLE001 - skip unreadable files
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
    """Severity for the event, and whether it may raise an alert.

    Every detection gets a severity. Raising an alert also needs enough confidence, margin,
    agreement and quality, so a low-confidence Gunshot the models disagree on is stored but
    does not alert.
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

    # FR xliv/xlv: horns and animals are stored but don't alert. Checked after escalation so
    # loud background noise escalated to Medium (FR l) still alerts.
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
    """Check only the conditions this record has measurements for."""
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
    """Leaves unknown template keys as {key} instead of raising."""

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

    Only a fixed set of keys is supported, so the admin-editable file can't run code.
    Unknown keys evaluate to False.
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
        else:  # unknown key
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
    """Review conditions that fired, each with its reason and recommended action."""
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

    # FR lxxiv: a likely copy goes to review, it is never merged or dropped. A near_duplicate
    # condition in manual_review_conditions.json overrides this default.
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
                        f"({audio_id}). It looks like a re-encoded, trimmed or "
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
        # A probable copy still needs a look, however confident the models are.
        and not near_duplicate
    )

    return {
        "required": bool(matched),
        "matched": [m["id"] for m in matched],
        "findings": matched,
        "priority": top_priority if matched else None,
        "conditions_evaluated": len(store.review_conditions()),
        "clean_result": clean,
        # Only a clean result with no conditions stays out of the queue.
        "never_auto_review_ok": (not clean) or not matched,
    }


# The pipeline

def _extractor_for_bundle(model_dir: Path) -> Any:
    """Build the feature extractor described by the bundle's feature_config.json."""
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
    """Both models plus the preprocessing they share, loaded once."""

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

    ``persist`` is optional so tests and benchmarks can run without a database.
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

        There is no fallback to a single model.
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

        # Both models must use the same class names or the comparison means nothing.
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
        """Run one dummy analysis so librosa/numba warm-up doesn't hit the first request."""
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
        except Exception as exc:  # warm-up failure shouldn't stop start-up
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

        Bad audio doesn't raise: the result has ``ok=False``, the reason and any quality
        measurements. ``meta`` is stored but doesn't affect the prediction.
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
            # Keep the original rate/channels for the report.
            "source_sample_rate": (preprocessed.preprocessing or {}).get("source_sample_rate"),
            "source_channels": (preprocessed.preprocessing or {}).get("source_channels"),
            "source_bit_depth": (preprocessed.preprocessing or {}).get("source_bit_depth"),
            "n_segments": len(preprocessed.segments or []),
            "source_path": str(preprocessed.source_path) if preprocessed.source_path else None,
            # sha256 matches identical bytes, the fingerprint similar sound. The caller does
            # the exact-duplicate check since it needs the database.
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
        # The older one-line verdict is kept alongside the condition-based one.
        legacy_required, legacy_reason = requires_manual_review(comparison, thresholds)
        record["comparison"]["requires_review_quick_check"] = bool(legacy_required)
        record["comparison"]["quick_check_reason"] = legacy_reason

        # Severity, alert eligibility and repeated-detection confirmation.
        t0 = time.perf_counter()
        # The Python model drives the decision; the TM result grades it (SRS Step 11).
        primary = py_result.predicted_class
        rule = self.store.rule_for_class(primary)
        confirmation = self.tracker.observe(
            class_name=primary,
            agreed=bool(comparison.classes_agree),
            quality=str(record["quality"].get("verdict")),
            confidence=float(py_result.confidence),
            at=at,
            source=origin,
            # An upload has no next window to wait for, so it uses required_for_uploads and
            # relies on the other checks (agreement, quality, confidence, margin).
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
        # Candidates come from the caller (the database).
        duplicate_settings = duplicate_config(thresholds)
        near_duplicate = None
        candidates = list(meta.get("near_duplicate_candidates") or ())
        exclude = {meta.get("audio_id")} if meta.get("audio_id") else ()
        if duplicate_settings["enabled"] and record.get("audio", {}).get("fingerprint"):
            if candidates and len(candidates[0]) == 3:
                # With stored file paths we can run the two-stage check; otherwise only
                # the fingerprint.
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
            # `checked` tells "no match found" apart from "didn't look".
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
        # A critical class without agreement always goes to review, even if no condition
        # named it.
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
                # Return the analysis anyway, but record that saving failed.
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
        """Analyse uploaded bytes, via a temp file so FFmpeg can read them."""
        from src.inference.contract import AudioSource

        tmp_dir = REPO_ROOT / "data" / "tmp" / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Name the file by its hash so identical uploads can't collide.
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
    """One model's output with the confidence for every class."""
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
    """Likely Valid, Likely Invalid or Manual Review Required."""
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


# One pipeline per process, built by the app factory

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
    """Pipeline status for /api/health and the admin dashboard."""
    if _PIPELINE is None:
        return {"ready": False, "reason": "pipeline not initialised"}
    return {
        "ready": True,
        "models": _PIPELINE.models.describe(),
        "warmed": _PIPELINE.warmed,
        "tracker": _PIPELINE.tracker.state(),
        "config": _PIPELINE.store.snapshot().to_dict(),
    }
