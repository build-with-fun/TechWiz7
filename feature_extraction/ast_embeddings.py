"""Audio Spectrogram Transformer (AST) embeddings for the Python model.

CNN14 embeddings plateaued at about 0.84 validation accuracy whatever classifier sat on
top (26 Sep experiments in documentation/MODEL_EVALUATION.md). AST (Gong et al., 2021),
fine-tuned on AudioSet, separates our confusable pairs better. It is used here the same
way as CNN14: frozen, as a feature extractor, with a small classifier trained on our
2,100 training recordings.

Weights: MIT/ast-finetuned-audioset-10-10-0.4593 on Hugging Face (BSD-3-Clause), pinned
to one revision and downloaded once into the Hugging Face cache (~350 MB).

Features per clip: the pooled [CLS]/[DIST] token (768), the mean of the patch tokens
(768), and AST's own 527 AudioSet scores (sigmoid), averaged over 10.24 s chunks. The
classifier decides which of them matter; the AudioSet scores are inputs to it, never a
decision on their own.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
EMBEDDING_VERSION = "ast-audioset-10-10-emb-1.0.0"
SAMPLE_RATE = 16000
CHUNK = int(10.24 * SAMPLE_RATE)   # the 1024 frames AST's position embeddings cover
MIN_TAIL = SAMPLE_RATE             # a tail under 1 s joins the previous chunk (AST's 1024 frames then trim it)
POOLED_DIM, TOKENS_DIM, AUDIOSET_DIM = 768, 768, 527
EMBEDDING_DIM = POOLED_DIM + TOKENS_DIM + AUDIOSET_DIM


def chunks(y: np.ndarray) -> list[np.ndarray]:
    """Split a waveform into <=10.24 s pieces; a clip shorter than that is one piece."""
    if y.size <= CHUNK:
        return [y]
    starts = list(range(0, y.size, CHUNK))
    if y.size - starts[-1] < MIN_TAIL:
        starts.pop()
    pieces = [y[s:s + CHUNK] for s in starts[:-1]]
    pieces.append(y[starts[-1]:])
    return pieces


class AstEmbedder:
    """Turns a mono waveform into a fixed-length AST feature vector (eval mode, no grad)."""

    _lock = threading.Lock()

    def __init__(self, *, threads: int | None = None) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        try:
            self.frontend = ASTFeatureExtractor.from_pretrained(AST_MODEL, revision=AST_REVISION)
            self.net = ASTForAudioClassification.from_pretrained(AST_MODEL, revision=AST_REVISION).eval()
        except OSError as exc:
            raise FileNotFoundError(
                f"AST weights {AST_MODEL}@{AST_REVISION[:8]} are not in the Hugging Face cache "
                "and could not be downloaded. Run `python tools/fetch_pretrained.py --ast` "
                "once with network access."
            ) from exc
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
                out = self.net.audio_spectrogram_transformer(x["input_values"])
                pooled = out.pooler_output[0]
                tokens = out.last_hidden_state[0, 2:].mean(dim=0)
                scores = torch.sigmoid(self.net.classifier(out.pooler_output))[0]
                rows.append(torch.cat([pooled, tokens, scores]).numpy())
        return np.mean(rows, axis=0).astype(np.float64)


_shared: AstEmbedder | None = None
_shared_lock = threading.Lock()


def shared_embedder() -> AstEmbedder:
    """One AST per process; loading it takes a few seconds."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AstEmbedder(threads=int(os.environ.get("SST_TORCH_THREADS", "0")) or None)
        return _shared


def embedding_columns() -> list[str]:
    return ([f"ast_pooled_{i:03d}" for i in range(POOLED_DIM)]
            + [f"ast_tokens_{i:03d}" for i in range(TOKENS_DIM)]
            + [f"ast_audioset_{i:03d}" for i in range(AUDIOSET_DIM)])


class AstFeatureExtractor:
    """Feature extractor for bundles whose feature_version is the AST embedding version.

    Receives the same PreprocessedAudio the GTM model receives, like the CNN14 extractor.
    """

    feature_version = EMBEDDING_VERSION

    def __init__(self, embedder: AstEmbedder | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> AstEmbedder:
        if self._embedder is None:
            self._embedder = shared_embedder()
        return self._embedder

    def extract(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        samples = getattr(preprocessed_or_samples, "samples", preprocessed_or_samples)
        rate = sample_rate or getattr(preprocessed_or_samples, "sample_rate", None) or SAMPLE_RATE
        return self.embedder.embed(samples, int(rate))
