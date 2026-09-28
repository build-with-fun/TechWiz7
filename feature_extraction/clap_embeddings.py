"""CLAP audio embeddings, one of the three inputs of the served Python model.

CLAP (Wu et al., 2023) learned to match audio with text, so it makes different mistakes
from the AudioSet networks. It stays frozen. Weights: laion/larger_clap_general on Hugging
Face (Apache-2.0), pinned revision, ~780 MB.

Per clip: the audio encoder's 1,024-value pooled output, averaged over 10 s chunks. We cut
long clips ourselves, so CLAP never crops at random and the result is repeatable.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

CLAP_MODEL = "laion/larger_clap_general"
CLAP_REVISION = "ada0c23a36c4e8582805bb38fec3905903f18b41"
EMBEDDING_VERSION = "clap-larger-general-emb-1.0.0"
SAMPLE_RATE = 48000
CHUNK = 10 * SAMPLE_RATE           # CLAP's input length
MIN_TAIL = SAMPLE_RATE             # a tail under 1 s is merged into the previous chunk
EMBEDDING_DIM = 1024


def chunks(y: np.ndarray) -> list[np.ndarray]:
    """Split a waveform into pieces of at most 10 s."""
    if y.size <= CHUNK:
        return [y]
    starts = list(range(0, y.size, CHUNK))
    if y.size - starts[-1] < MIN_TAIL:
        starts.pop()
    pieces = [y[s:s + CHUNK] for s in starts[:-1]]
    pieces.append(y[starts[-1]:starts[-1] + CHUNK])
    return pieces


class ClapEmbedder:
    """Mono waveform -> fixed-length CLAP feature vector."""

    _lock = threading.Lock()

    def __init__(self, *, threads: int | None = None) -> None:
        import torch
        from transformers import ClapFeatureExtractor, ClapModel

        try:
            self.frontend = ClapFeatureExtractor.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION)
            model = ClapModel.from_pretrained(CLAP_MODEL, revision=CLAP_REVISION)
        except OSError as exc:
            raise FileNotFoundError(
                f"CLAP weights {CLAP_MODEL}@{CLAP_REVISION[:8]} are not in the Hugging Face "
                "cache and could not be downloaded. Run `python tools/fetch_pretrained.py "
                "--clap` once with network access."
            ) from exc
        # Keep only the audio half.
        self.net = model.audio_model.eval()
        del model
        if threads:
            torch.set_num_threads(threads)

    def embed(self, samples: Any, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        import torch

        y = np.asarray(samples, dtype=np.float32).ravel()
        if sample_rate != SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        rows = []
        with self._lock, torch.inference_mode():
            for piece in chunks(y):
                x = self.frontend(piece, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                out = self.net(input_features=x["input_features"], is_longer=x["is_longer"])
                rows.append(out.pooler_output[0].numpy())
        return np.mean(rows, axis=0).astype(np.float64)


_shared: ClapEmbedder | None = None
_shared_lock = threading.Lock()


def shared_embedder() -> ClapEmbedder:
    """Shared CLAP model (loading takes a few seconds)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ClapEmbedder(threads=int(os.environ.get("SST_TORCH_THREADS", "0")) or None)
        return _shared


def embedding_columns() -> list[str]:
    return [f"clap_pooled_{i:04d}" for i in range(EMBEDDING_DIM)]


class ClapFeatureExtractor:
    """Feature extractor for bundles that use CLAP embeddings."""

    feature_version = EMBEDDING_VERSION

    def __init__(self, embedder: ClapEmbedder | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> ClapEmbedder:
        if self._embedder is None:
            self._embedder = shared_embedder()
        return self._embedder

    def extract(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        samples = getattr(preprocessed_or_samples, "samples", preprocessed_or_samples)
        rate = sample_rate or getattr(preprocessed_or_samples, "sample_rate", None) or SAMPLE_RATE
        return self.embedder.embed(samples, int(rate))
