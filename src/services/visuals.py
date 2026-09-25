"""Build cached waveform peaks and a mel-spectrogram visualization for event detail.

The display transform covers the whole recording and is intentionally separate from
the short browser-FFT window consumed by the Teachable Machine model.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

# How many waveform columns the client draws. 1200 matches the canvas width in
# templates/event_detail.html; one peak pair per pixel keeps the drawing trivial.
PEAK_COLUMNS = 1200

_CACHE_LOCK = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}


def _cache_path(storage_root: Path, audio_id: str, stored_path: str) -> Path:
    """Cache file sits beside the recording's directory root: visuals/<audio_id>.json."""
    return Path(storage_root) / "visuals" / f"{audio_id or 'unknown'}.json"


def _load_cached(cache_file: Path) -> dict[str, Any] | None:
    try:
        return json.loads(cache_file.read_text())
    except (OSError, ValueError):
        return None


def _store_cached(cache_file: Path, payload: dict[str, Any]) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
    except OSError:
        pass  # a failed cache write must never fail the request


def _waveform_peaks(samples: np.ndarray, columns: int) -> list[float]:
    """min/max peak pairs, normalised to [-1, 1] against the loudest peak."""
    if samples.size == 0:
        return []
    edges = np.linspace(0, samples.size, columns + 1, dtype=np.int64)
    peaks: list[float] = []
    for i in range(columns):
        chunk = samples[edges[i]: edges[i + 1]]
        if chunk.size == 0:
            peaks.append(0.0)
            continue
        peaks.append(round(float(np.max(np.abs(chunk))), 4))
    peak = max(peaks) or 1.0
    return [round(p / peak, 4) for p in peaks]


def build_visuals(
    storage: Any,
    stored_path: str,
    *,
    event_id: int,
    fallback_duration: float | None = None,
    fallback_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Return ``{duration_sec, sample_rate, peaks, spectrogram: {n_mels, n_frames, data}}``.

    ``peaks`` is a list of normalised non-negative peak amplitudes (length
    :data:`PEAK_COLUMNS`). ``data`` is the mel spectrogram as a flat row-major float list
    (n_mels rows x n_frames columns), normalised to 0..1 so the client needs no scaling.
    """
    path = storage.resolve(stored_path)
    cache_file = _cache_path(storage.root, str(event_id), stored_path)

    with _CACHE_LOCK:
        cached = _load_cached(cache_file)
    if cached is not None:
        return cached

    import librosa

    # Enough headroom for the whole clip; uploads are <= 30 s and live windows ~2 s.
    y, sr = librosa.load(path, sr=None, mono=True)

    peaks = _waveform_peaks(y.astype("float32"), PEAK_COLUMNS)

    from src.services.config import get_store
    from src.inference.gtm_predictor import GtmFrontendConfig

    frontend = (get_store().thresholds() or {}).get("gtm_frontend", {}) or {}
    cfg = GtmFrontendConfig(
        sample_rate=int(frontend.get("sample_rate", sr)),
        window_sec=float(frontend.get("window_sec", fallback_duration or (y.size / sr))),
        n_mels=int(frontend.get("n_mels", 64)),
        n_frames=int(frontend.get("n_frames", 96)),
        fmin=float(frontend.get("fmin", 0.0)),
        fmax=float(frontend.get("fmax", sr / 2.0)),
        normalize_mode="max",
        frontend_id="visuals",
        source="visuals",
    )

    mel = _mel(y, cfg)
    peak = float(mel.max()) or 1.0
    mel = (mel / peak).astype("float32")

    payload = {
        "event_id": event_id,
        "duration_sec": round(float(y.size / sr), 3),
        "sample_rate": int(sr),
        "peaks": peaks,
        "spectrogram": {
            "n_mels": int(mel.shape[0]),
            "n_frames": int(mel.shape[1]),
            "data": [round(float(v), 4) for v in mel.ravel().tolist()],
        },
    }
    with _CACHE_LOCK:
        _store_cached(cache_file, payload)
    return payload


def _mel(y: np.ndarray, cfg: Any) -> np.ndarray:
    """Mel spectrogram without the GTM window cropping -- draw the WHOLE recording."""
    import librosa

    hop = max(1, y.size // cfg.n_frames) if y.size else 1
    mel = librosa.feature.melspectrogram(
        y=y.astype("float64"),
        sr=cfg.sample_rate,
        n_fft=1024,
        hop_length=hop,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax or cfg.sample_rate / 2.0,
        power=2.0,
    )
    frames = mel.shape[1]
    if frames < cfg.n_frames:
        mel = np.pad(mel, ((0, 0), (0, cfg.n_frames - frames)))
    return mel[:, : cfg.n_frames]
