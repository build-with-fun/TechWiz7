"""CNN14 embeddings for the transfer-learning Python model.

The hand-made features in features.py stopped at about 0.70 test accuracy, mostly failing
on classes with similar spectra (door slam vs gunshot, scream vs shout). CNN14 from PANNs
(Kong et al., 2020), pretrained on AudioSet, handles those better, so we take its
2048-value penultimate layer and train a small classifier on it.

AudioSet is YouTube audio and our clips come from Freesound datasets plus our own, so the
pretraining data doesn't include our test set. Weights (~300 MB) are not in Git;
tools/fetch_pretrained.py downloads and checks them. Only the backbone and fc1 are
loaded, not the AudioSet classification head.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDING_VERSION = "panns-cnn14-16k-emb-1.0.0"
EMBEDDING_DIM = 2048

# Front-end settings of the 16 kHz checkpoint. They must match it exactly.
SAMPLE_RATE = 16000
WINDOW = 512
HOP = 160
MEL_BINS = 64
FMIN = 50.0
FMAX = 8000.0

CHECKPOINT_NAME = "Cnn14_16k_mAP=0.438.pth"
CHECKPOINT_URL = (
    "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
)
# SHA-256 of our copy, which matched Zenodo's md5 (362fc5ff18f1d6ad2f6d464b45893f2c).
CHECKPOINT_MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
CHECKPOINT_SHA256 = "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"

# CNN14's pooling needs at least ~0.32 s of input; pad to one second.
MIN_SAMPLES = SAMPLE_RATE


def default_checkpoint_path() -> Path:
    override = os.environ.get("SST_PANNS_CHECKPOINT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "sonicsentinel" / CHECKPOINT_NAME


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_network():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.bn2 = nn.BatchNorm2d(cout)

        def forward(self, x, pool):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return F.avg_pool2d(x, kernel_size=pool)

    class Cnn14Embedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            import librosa

            mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=WINDOW, n_mels=MEL_BINS,
                                      fmin=FMIN, fmax=FMAX)
            self.register_buffer("mel_basis", torch.tensor(mel.T, dtype=torch.float32))
            self.register_buffer("window", torch.hann_window(WINDOW, periodic=True))
            self.bn0 = nn.BatchNorm2d(MEL_BINS)
            widths = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]
            for i, (cin, cout) in enumerate(widths, start=1):
                setattr(self, f"conv_block{i}", ConvBlock(cin, cout))
            self.fc1 = nn.Linear(2048, 2048)

        def log_mel(self, wave):
            # Same as the checkpoint's STFT: centred, reflect-padded, periodic Hann,
            # power, then 10*log10 with a 1e-10 floor.
            spec = torch.stft(wave, n_fft=WINDOW, hop_length=HOP, win_length=WINDOW,
                              window=self.window, center=True, pad_mode="reflect",
                              return_complex=True)
            power = spec.real ** 2 + spec.imag ** 2          # (batch, freq, time)
            mel = torch.matmul(power.transpose(1, 2), self.mel_basis)
            return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))

        def forward(self, wave):
            x = self.log_mel(wave).unsqueeze(1)              # (batch, 1, time, mel)
            x = self.bn0(x.transpose(1, 3)).transpose(1, 3)  # bn0 normalises per mel band
            for i in range(1, 6):
                x = getattr(self, f"conv_block{i}")(x, (2, 2))
            x = self.conv_block6(x, (1, 1))
            x = x.mean(dim=3)                                # collapse frequency
            # Max + mean over time: max keeps short impulses, mean helps sustained sounds.
            x = x.max(dim=2).values + x.mean(dim=2)
            return F.relu(self.fc1(x))

    return Cnn14Embedder()


class PannsEmbedder:
    """16 kHz mono waveform -> 2048-d CNN14 embedding."""

    _lock = threading.Lock()

    def __init__(self, checkpoint: str | Path | None = None, *, threads: int | None = None) -> None:
        import torch

        path = Path(checkpoint) if checkpoint else default_checkpoint_path()
        if not path.exists():
            raise FileNotFoundError(
                f"PANNs checkpoint not found at {path}. Run "
                "`python tools/fetch_pretrained.py` (downloads ~300 MB once)."
            )
        if CHECKPOINT_SHA256 and sha256_of(path) != CHECKPOINT_SHA256:
            raise ValueError(f"{path} does not match the expected SHA-256; re-download it")

        net = _build_network()
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        wanted = net.state_dict()
        picked = {k: v for k, v in state.items() if k in wanted}
        missing = [k for k in wanted if k not in picked and k not in ("mel_basis", "window")]
        if missing:
            raise ValueError(f"checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
        net.load_state_dict(picked, strict=False)
        net.eval()
        self.net = net
        self.checkpoint = path
        if threads:
            torch.set_num_threads(threads)

    def embed(self, samples: Any, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        import torch

        y = np.asarray(samples, dtype=np.float32).ravel()
        if sample_rate != SAMPLE_RATE:
            import librosa

            y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        if y.size < MIN_SAMPLES:
            # Centre-pad so short sounds have silence on both sides, like the training clips.
            total = MIN_SAMPLES - y.size
            y = np.pad(y, (total // 2, total - total // 2))
        # One request at a time through torch.
        with self._lock, torch.inference_mode():
            out = self.net(torch.from_numpy(y).unsqueeze(0))
        return out.squeeze(0).numpy().astype(np.float64)


_shared: PannsEmbedder | None = None
_shared_lock = threading.Lock()


def shared_embedder() -> PannsEmbedder:
    """Shared embedder (loading takes a couple of seconds)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = PannsEmbedder()
        return _shared


def embedding_columns() -> list[str]:
    return [f"panns_{i:04d}" for i in range(EMBEDDING_DIM)]


class EmbeddingFeatureExtractor:
    """Feature extractor for bundles that use CNN14 embeddings.

    Same interface as FeatureExtractor, and gets the same PreprocessedAudio as the TM model.
    """

    feature_version = EMBEDDING_VERSION

    def __init__(self, embedder: PannsEmbedder | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> PannsEmbedder:
        if self._embedder is None:
            self._embedder = shared_embedder()
        return self._embedder

    def extract(self, preprocessed_or_samples: Any, sample_rate: int | None = None) -> np.ndarray:
        samples = getattr(preprocessed_or_samples, "samples", preprocessed_or_samples)
        rate = sample_rate or getattr(preprocessed_or_samples, "sample_rate", None) or SAMPLE_RATE
        return self.embedder.embed(samples, int(rate))
