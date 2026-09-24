"""Segment enumeration, the on-disk Mel cache, and train-only normalisation statistics.

Owner: taha.  Written to satisfy nadia's four conditions on the ``(128, 94)`` log-mel
contract (agreed 2026-09-23), each of which is cheap now and expensive later:

1. **Fixed ``ref=1.0`` in ``power_to_db``** -- done in
   :func:`feature_extraction.features.segment_logmel`; absolute level is real signal here
   and per-clip max-normalisation would destroy it.  Gain-jitter augmentation is hers.
2. **Train-only normalisation statistics** -- :class:`MelNormalizer`.  ``fit`` takes a
   ``split`` argument with no default and *refuses any value other than ``"train"``*, so
   training-split statistics can never leak into validation or test data (SRS integrity
   rules).  The statistics are saved beside the model so inference uses the same numbers.
3. **A reproducible cache** -- :class:`MelCache` writes
   ``data/features/mel/{audio_id}/{start_ms}-{end_ms}.npy`` plus an index CSV carrying
   ``segment_start_sec``/``segment_end_sec``, so any tensor can be traced back to the
   segment ID that produced it.
4. **Segment enumeration is public** -- :func:`enumerate_segments` returns the timeline the
   tensors correspond to.  The app aggregates over it; nothing is hidden inside the tensor
   call, and row *i* of the tensor stack is row *i* of the enumeration by construction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from audio_preprocessing import config as cfg_mod
from audio_preprocessing.transforms import segment_bounds, to_mono

MEL_INDEX_COLUMNS = (
    "audio_id",
    "segment_index",
    "segment_start_sec",
    "segment_end_sec",
    "start_ms",
    "end_ms",
    "duration_sec",
    "n_mels",
    "n_frames",
    "feature_version",
    "source_path",
    "content_hash",
)


# --------------------------------------------------------------------------------------
# 4. Segment enumeration
# --------------------------------------------------------------------------------------

def enumerate_segments(
    preprocessed_or_samples: Any,
    sample_rate: int | None = None,
    *,
    feature_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The segment timeline, as data the app can aggregate over.

    Returns one dict per segment::

        {"index": 0, "start_sec": 0.0, "end_sec": 3.0, "start_ms": 0, "end_ms": 3000,
         "duration_sec": 3.0, "n_samples": 48000}

    This is the same enumeration ``extract_mel_segments`` and
    ``FeatureExtractor.extract_segments`` iterate, so the *i*-th tensor and the *i*-th
    timestamp can never disagree.  The per-segment decision in the UI, the event timeline
    and the ``segment_id`` in the report all index into this list.
    """
    conf = _conf(feature_config)
    y, sr, _ = _signal(preprocessed_or_samples, sample_rate)
    if y.size == 0:
        return []
    seg_seconds = float(conf["segment_duration_sec"])
    out: list[dict[str, Any]] = []
    for i, (s, e) in enumerate(segment_bounds(y.size, sr, seg_seconds, mode="cover")):
        start_sec = round(s / sr, 6)
        end_sec = round(e / sr, 6)
        out.append({
            "index": i,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_ms": int(round(start_sec * 1000)),
            "end_ms": int(round(end_sec * 1000)),
            "duration_sec": round((e - s) / sr, 6),
            "n_samples": int(e - s),
        })
    return out


def _conf(feature_config: dict[str, Any] | None) -> dict[str, Any]:
    """Segment length and rate for the enumeration -- same rule as ``features._fc``.

    A caller-supplied config carrying its own ``_audio_block`` keeps its own segment
    length; otherwise the repository audio block applies.  Without this the cache and the
    extractor could disagree about where segment *i* starts, and row *i* of the tensor
    stack would no longer correspond to row *i* of the enumeration.
    """
    conf = dict(feature_config) if feature_config is not None else cfg_mod.feature_config()
    audio = conf.get("_audio_block")
    if not isinstance(audio, dict):
        audio = cfg_mod.audio_config()
    conf["sample_rate"] = int(audio["target_sample_rate"])
    conf["segment_duration_sec"] = float(audio["segment_duration_sec"])
    return conf


def _signal(source: Any, sample_rate: int | None) -> tuple[np.ndarray, int, list]:
    from feature_extraction.features import _as_signal

    return _as_signal(source, sample_rate)


# --------------------------------------------------------------------------------------
# Audio identity
# --------------------------------------------------------------------------------------

def audio_id_for(source: Any) -> str:
    """Stable identifier for a recording, used as the cache directory name.

    File-backed audio hashes the file *contents* (via lorena's ``audio_fingerprint`` when
    importable, so duplicate detection and the cache agree).  In-memory audio hashes the
    samples and the rate.  Same recording, same id, every run -- which is what makes the
    cache reproducible rather than merely fast.
    """
    path = getattr(source, "path", None)
    if path is not None:
        try:
            from src.inference.contract import audio_fingerprint

            return str(audio_fingerprint(source))[:32]
        except Exception:
            pass
        from audio_preprocessing.io import sha256_file

        return f"file-{sha256_file(path)[:32]}"
    samples = getattr(source, "samples", None)
    rate = getattr(source, "sample_rate", None)
    if samples is None:
        samples, rate = source, 16000
    arr = to_mono(np.asarray(samples, dtype=np.float32))
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    h.update(f"@{int(rate)}".encode())
    return f"mem-{h.hexdigest()[:32]}"


# --------------------------------------------------------------------------------------
# 3. The Mel cache
# --------------------------------------------------------------------------------------

@dataclass
class MelCacheEntry:
    audio_id: str
    segment_index: int
    segment_start_sec: float
    segment_end_sec: float
    start_ms: int
    end_ms: int
    duration_sec: float
    n_mels: int
    n_frames: int
    feature_version: str
    source_path: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MelCache:
    """Content-addressed store of log-mel tensors, in the layout nadia specified.

    Layout::

        <root>/<audio_id>/<start_ms>-<end_ms>.npy
        <root>/index.csv

    ``.npy`` rather than ``.npz``: one file per segment means a single segment can be
    loaded without reading the others, which the live path and the per-segment UI both
    need, and the filename *is* the segment id.

    The index CSV is rewritten in sorted order on every save, so it is deterministic and
    diff-able in git -- adding a recording must not reshuffle unrelated rows.
    """

    def __init__(self, root: str | Path = "data/features/mel", *, feature_version: str | None = None) -> None:
        from feature_extraction.features import FEATURE_SCHEMA_VERSION

        self.root = Path(root)
        self.feature_version = feature_version or FEATURE_SCHEMA_VERSION

    # -- paths -------------------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / "index.csv"

    def audio_dir(self, audio_id: str) -> Path:
        return self.root / audio_id

    def segment_path(self, audio_id: str, start_ms: int, end_ms: int) -> Path:
        return self.audio_dir(audio_id) / f"{int(start_ms)}-{int(end_ms)}.npy"

    # -- single-segment IO -------------------------------------------------------------
    def has(self, audio_id: str, start_ms: int, end_ms: int) -> bool:
        return self.segment_path(audio_id, start_ms, end_ms).exists()

    def save_segment(self, audio_id: str, start_ms: int, end_ms: int, tensor: np.ndarray) -> Path:
        path = self.segment_path(audio_id, start_ms, end_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(np.asarray(tensor, dtype=np.float32)))
        return path

    def load_segment(self, audio_id: str, start_ms: int, end_ms: int) -> np.ndarray:
        path = self.segment_path(audio_id, start_ms, end_ms)
        if not path.exists():
            raise FileNotFoundError(f"no cached mel tensor at {path}")
        return np.load(path)

    def load_by_id(self, audio_id: str, segment_index: int) -> np.ndarray:
        """Load a segment's tensor from its ``audio_id`` + ``segment_index`` (nadia: "I need
        to reproduce a segment from its ID")."""
        for row in self.load_index():
            if row["audio_id"] == audio_id and int(row["segment_index"]) == int(segment_index):
                return self.load_segment(audio_id, int(row["start_ms"]), int(row["end_ms"]))
        raise KeyError(f"{audio_id} has no segment {segment_index} in {self.index_path}")

    # -- whole-recording IO -----------------------------------------------------------
    def save_recording(self, source: Any, tensors: np.ndarray, *, source_path: str | None = None) -> list[MelCacheEntry]:
        """Write every segment tensor plus its index rows, and return what was written.

        ``tensors`` must be ``(n_segments, n_mels, n_frames)`` and in the same order as
        ``enumerate_segments`` -- enforced, because a silent off-by-one here would train a
        model on mismatched labels and nothing downstream would notice.
        """
        segments = enumerate_segments(source)
        arr = np.asarray(tensors, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"tensors must be (n_segments, n_mels, n_frames), got {arr.shape}")
        if arr.shape[0] != len(segments):
            raise ValueError(
                f"tensor count {arr.shape[0]} does not match the segment enumeration "
                f"({len(segments)} segments for this recording)"
            )
        audio_id = audio_id_for(source)
        path_label = source_path if source_path is not None else str(getattr(source, "path", "<memory>"))
        entries: list[MelCacheEntry] = []
        for seg, tensor in zip(segments, arr, strict=True):
            self.save_segment(audio_id, seg["start_ms"], seg["end_ms"], tensor)
            entries.append(MelCacheEntry(
                audio_id=audio_id,
                segment_index=int(seg["index"]),
                segment_start_sec=float(seg["start_sec"]),
                segment_end_sec=float(seg["end_sec"]),
                start_ms=int(seg["start_ms"]),
                end_ms=int(seg["end_ms"]),
                duration_sec=float(seg["duration_sec"]),
                n_mels=int(arr.shape[1]),
                n_frames=int(arr.shape[2]),
                feature_version=self.feature_version,
                source_path=path_label,
                content_hash=_hash_tensor(tensor),
            ))
        self._upsert_index(entries)
        return entries

    def get_or_compute(self, source: Any, *, force: bool = False) -> tuple[np.ndarray, list[dict[str, Any]], bool]:
        """Return ``(tensors, segments, from_cache)`` for a recording, computing on a miss.

        The cache is validated against the *index* (feature version and tensor shape), not
        merely against the file's existence: a stale tensor from an earlier feature version
        must be recomputed, not silently reused.
        """
        from feature_extraction.features import extract_mel_segments

        segments = enumerate_segments(source)
        audio_id = audio_id_for(source)
        if not force and segments:
            indexed = {int(r["segment_index"]): r for r in self.load_index() if r["audio_id"] == audio_id}
            if len(indexed) == len(segments) and all(
                indexed[i]["feature_version"] == self.feature_version
                and int(indexed[i]["start_ms"]) == segments[i]["start_ms"]
                for i in range(len(segments))
            ):
                try:
                    stack = np.stack([self.load_by_id(audio_id, i) for i in range(len(segments))], axis=0)
                    return stack, segments, True
                except (FileNotFoundError, KeyError):
                    pass
        tensors = extract_mel_segments(source)
        self.save_recording(source, tensors)
        return tensors, segments, False

    # -- index -------------------------------------------------------------------------
    def load_index(self) -> list[dict[str, Any]]:
        """The whole index.  Returns ``[]`` when the cache has never been written."""
        if not self.index_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.index_path.open(newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if value is None or value == "":
                        row[key] = value
                        continue
                    if key in {"segment_index", "start_ms", "end_ms", "n_mels", "n_frames"}:
                        row[key] = int(value)
                    elif key in {"segment_start_sec", "segment_end_sec", "duration_sec"}:
                        row[key] = float(value)
                    else:
                        row[key] = value
                rows.append(row)
        return rows

    def _upsert_index(self, entries: Sequence[MelCacheEntry]) -> None:
        """Replace this audio_id's rows and rewrite the whole index in deterministic order."""
        existing = [r for r in self.load_index() if r["audio_id"] != entries[0].audio_id] if entries else self.load_index()
        merged = existing + [e.to_dict() for e in entries]
        merged.sort(key=lambda r: (str(r["audio_id"]), int(r["segment_index"])))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(MEL_INDEX_COLUMNS))
            writer.writeheader()
            for row in merged:
                writer.writerow({k: row.get(k, "") for k in MEL_INDEX_COLUMNS})

    def stats(self) -> dict[str, Any]:
        """How much is cached, for the diagnostics page."""
        rows = self.load_index()
        return {
            "root": str(self.root),
            "index": str(self.index_path),
            "recordings": len({r["audio_id"] for r in rows}),
            "segments": len(rows),
            "feature_version": self.feature_version,
        }


def _hash_tensor(tensor: np.ndarray) -> str:
    """Content hash of one tensor, so a corrupted cache file is detectable."""
    return hashlib.sha256(np.ascontiguousarray(tensor, dtype="<f4").tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 2. Train-only normalisation
# --------------------------------------------------------------------------------------

@dataclass
class MelNormalizer:
    """Per-band z-score statistics for the log-mel tensor, **fitted on the train split only**.

    ``fit`` requires the caller to state which split it is fitting on and refuses anything
    other than ``"train"``.  That is deliberate: the single easiest way to fail the SRS
    integrity checks is to compute normalisation statistics over the whole dataset, and a
    convention nobody can enforce is not a control.  A kwarg with no default plus a guard
    is.

    The statistics are saved as JSON next to the model so inference reproduces training
    exactly; loading a normaliser fitted on a different feature version is refused.
    """

    mean: np.ndarray
    std: np.ndarray
    n_samples: int = 0
    feature_version: str = ""
    fitted_on: str = "train"

    @classmethod
    def fit(
        cls,
        tensors: Iterable[np.ndarray],
        *,
        split: str,
        feature_version: str | None = None,
    ) -> "MelNormalizer":
        if split != "train":
            raise ValueError(
                f"MelNormalizer.fit may only be called with split='train' (got {split!r}); "
                "fitting on validation or test data leaks information the SRS forbids"
            )
        from feature_extraction.features import FEATURE_SCHEMA_VERSION

        stack = [np.asarray(t, dtype=np.float64)[None, ...] if np.asarray(t).ndim == 2 else np.asarray(t, dtype=np.float64)
                 for t in tensors]
        if not stack:
            raise ValueError("cannot fit MelNormalizer on zero tensors")
        data = np.concatenate(stack, axis=0)
        if data.ndim != 3:
            raise ValueError(f"expected (n, n_mels, n_frames) tensors, got {data.shape}")
        # Statistics per mel band, pooled over frames and segments: the band axis is the
        # physical one (Hz), so a per-band mean is meaningful and a per-frame mean is not.
        mean = data.mean(axis=(0, 2))
        std = data.std(axis=(0, 2))
        std = np.where(std < 1e-6, 1.0, std)
        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            n_samples=int(data.shape[0]),
            feature_version=feature_version or FEATURE_SCHEMA_VERSION,
        )

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.shape[-2] != self.mean.shape[0]:
            raise ValueError(
                f"normaliser was fitted on {self.mean.shape[0]} mel bands, tensor has {arr.shape[-2]}"
            )
        return ((arr - self.mean.reshape(-1, 1)) / self.std.reshape(-1, 1)).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "n_samples": self.n_samples,
            "feature_version": self.feature_version,
            "n_mels": int(self.mean.shape[0]),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path, *, expect_feature_version: str | None = None) -> "MelNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("fitted_on") != "train":
            raise ValueError(
                f"refusing to load normaliser fitted on {data.get('fitted_on')!r}; "
                "only train-fitted statistics may be used for inference"
            )
        if expect_feature_version and data.get("feature_version") != expect_feature_version:
            raise ValueError(
                f"normaliser feature_version {data.get('feature_version')!r} does not match "
                f"the running extractor {expect_feature_version!r}"
            )
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            n_samples=int(data.get("n_samples", 0)),
            feature_version=str(data.get("feature_version", "")),
            fitted_on=str(data.get("fitted_on", "train")),
        )
