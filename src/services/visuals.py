"""Waveform peaks and mel spectrogram of a stored recording (cached)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

# One peak per canvas pixel (templates/event_detail.html draws 1200 columns).
PEAK_COLUMNS = 1200

_CACHE_LOCK = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}


def _cache_path(storage_root: Path, audio_id: str, stored_path: str) -> Path:
    """visuals/<audio_id>.json next to the audio folder."""
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
        pass  # caching is optional


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
    """Normalised peaks (PEAK_COLUMNS long) and a 0..1 mel spectrogram as a flat list."""
    path = storage.resolve(stored_path)
    cache_file = _cache_path(storage.root, str(event_id), stored_path)

    with _CACHE_LOCK:
        cached = _load_cached(cache_file)
    if cached is not None:
        return cached

    import librosa

    # Uploads are at most 30 s, live windows about 2 s.
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
    """Mel spectrogram of the whole recording (no TM window cropping)."""
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


def report_figures(path: Path) -> dict[str, str]:
    """Waveform and spectrogram PNGs (base64) for the event report (FR lxix)."""
    import base64
    import io

    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = librosa.load(str(path), sr=16000, mono=True, duration=60.0)
    t = np.arange(y.size) / sr
    figures: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 2.2), dpi=110)
    ax.plot(t, y, linewidth=0.5, color="#1f6feb")
    ax.set(xlabel="Time (s)", ylabel="Amplitude", xlim=(0, max(t[-1] if t.size else 1, 0.1)))
    ax.set_title("Waveform")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    figures["waveform"] = base64.b64encode(buf.getvalue()).decode("ascii")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
    db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=110)
    img = ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
                    extent=(0, y.size / sr, 0, 64))
    ax.set(xlabel="Time (s)", ylabel="Mel band")
    ax.set_title("Mel spectrogram (dB relative to peak)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    figures["spectrogram"] = base64.b64encode(buf.getvalue()).decode("ascii")
    return figures
